myuser@staging-aizamd-routing-7d75dbd5b5-hfp7q:/app$ cat routing_service/services/notes.py 
"""Manages all services related to Note"""
import os
import re
from typing import Dict, List, Optional
import httpx
import logging
from enum import Enum
from pathlib import Path
import datetime as dt
from fastapi import HTTPException, status
from fastapi.encoders import jsonable_encoder
from routing_service.core.data_store import NoteStore, RecordNotFound
from routing_service.models import Note, EditNote, TitleUpdate, EHRNote
from routing_service.models.note import ICD10Code, EditImprovedNote
from datetime import datetime, timedelta
from routing_service.core.utils import (
    generate_id,
    create_soap_note,
    get_icd10_codes_aws,
    get_cached_icd10,
    find_unspecified_icd_child,
    filter_unique_codes,
    parse_icd10_codes,
    concatenate_transcript_text,
    post_messages,
    read_file,
    Diarization
)
from routing_service.core.data_store import (
    get_mongostore,
    NoteStore
)
from itertools import islice
import pytz

URL_CHATGPT = os.environ["LLM_SERVICE_URL"] + "/query-together"
LLM_URL = os.environ["LLM_SERVICE_URL"] + "/query-gemini-flash"


logger = logging.getLogger("routing_logger")
icd10_path = Path(__file__).parent.parent / "data" / "icd10cm-order-2024.txt"
with open(icd10_path, encoding="utf-8") as file:
    icd_codes = parse_icd10_codes(file)

class Platform(Enum):
    """The platform where the incoming request is coming from
    This is important to distinguish as we have differnt file formats 
    coming from mobile and web apps and we need to convert them accordingly"""

    mobile = 1
    web = 2


STT_URL: Dict[int, str] = {
    Platform.mobile: os.environ["STT_SERVICE_URL"] + "/transcribe/mobile",
    Platform.web: os.environ["STT_SERVICE_URL"] + "/transcribe/web",
}


async def diarize_audio(platform: Platform, audio) -> Diarization:
    """Post audio for diarization on respective STT endpoint"""

    timeout_seconds = 360
    async with httpx.AsyncClient() as client:
        files = {'audio': (audio.filename, await audio.read(), audio.content_type)}
        # Send the file to the stt endpoint for diarization
        diarized_audio = await client.post(STT_URL[platform], files=files, timeout=timeout_seconds)
    return diarized_audio


async def generate_title(service_url: str, diarization_output: Dict[str, Diarization]):
    """Generate automated note title using LLM"""
    sys_prompt_title = read_file("routing_service/core/prompts/title_prompt.txt")
    diarization_output = diarization_output['transcript']
    turns = 10  # no. of turns to pick from conversation for title generation since we don't need whole dialogue
    selective_dialogue = {'transcript': diarization_output[:turns]}
    concatenated_raw_note_text = concatenate_transcript_text(transcript_data=selective_dialogue)
    dialogue = f"""
        Dialogue:\n {concatenated_raw_note_text}
          
        Title: """
    llm_request = [{"role": "system", "content": sys_prompt_title}, {"role": "user", "content": dialogue}]
    return await post_messages(service_url, llm_request)


def extract_title(soap_note: str) -> str:
    """Extract the soap note title from the SOAP note.

    Args:
        soap_note (str): The SOAP note to extract the soap note title from.

    Returns:
        str: The soap note title from the SOAP note.
    """
    pattern = r"\*\*Chief Complaint\*\*: \"(.*?)\""
    # Use re.search to find the first match of the pattern in the text
    match = re.search(pattern, soap_note)
    if match:
        soap_note_title = match.group(1)
    else:
        soap_note_title = 'Edit this title'
    return soap_note_title


async def save_note(note_store: NoteStore, note_payload: Note, api_key) -> Note:
    """
    Save a note to the database.

    Args:
    note_store (NoteStore): The NoteStore instance to store the note.
    note_payload (Note): The note to be stored.
    api_key (str): The API key for authentication.

    Returns:
    Note: The saved note.

    Raises:
    HTTPException: If there is an error while storing the note.
    """
    payload_json = jsonable_encoder(note_payload)
    if not api_key:
        try:
             # We store diarization output making it robust to any server-side erros
            note_store.store_note(payload_json, note_payload.note_id)
        except Exception as exc:
            logging.exception("Something wrong while adding note on routing service.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Something went wrong on routing service.",
            ) from exc
    return note_payload
    
    
async def delete_old_ehr_fields():
    """
    This function deletes the 'ehr_data' field from the notes that were created more than 1 hour ago.
    It iterates over all the notes in the database, checks if the 'ehr_data' field exists, and if it does,
    it deletes the field and updates the note in the database.

    Returns:
        None
    """
    updated_ehr_records = 0
    note_store = NoteStore(get_mongostore())
    # Define the cutoff date (1 hour ago)
    cutoff_date = datetime.now(pytz.utc) - timedelta(hours=1)
    # Find records older than 1 hour
    records_to_update = note_store.find_notes(criteria={"date_created": {"$lt": cutoff_date}})
    for record in records_to_update:
        if 'ehr_data' in record:
            del record['ehr_data']
            updated_ehr_records += 1
            note_store.update_note(note_id=record["_id"], updated_data=record)
    logger.info(f"EHR information deleted from {updated_ehr_records} records at time {datetime.now(pytz.utc)}")


class NoteService:
    def __init__(self, note_store: NoteStore):
        """Initialize NoteService with a NoteStore instance."""
        self.note_store = note_store

    async def add_new(self, platform: Platform, user_id: str, api_key, audio, ehr_data: EHRNote = None) -> Note:
        """Add a new note and return the added note with modifications."""
        diarization_output = await diarize_audio(platform, audio)
        if diarization_output.status_code == status.HTTP_400_BAD_REQUEST:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=diarization_output.json().get("detail"),
            )
        else:
            diarization_output = diarization_output.json()
        note_payload = Note(
                    note_id=generate_id(),
                    user_id=user_id,
                    date_modified = dt.datetime.utcnow(),
                    raw_note=diarization_output,
                    date_created=dt.datetime.utcnow(),
                )
        note_payload.ehr_data = ehr_data or None
        payload_json = jsonable_encoder(note_payload)
        # if not api_key:
        try:
            # We store diarization output making it robust to any server-side erros
            self.note_store.store_note(payload_json, note_payload.note_id)
        except Exception as exc:
            logging.exception("Something wrong while adding note on routing service.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Something went wrong on routing service.",
            ) from exc
        return payload_json

    async def create_soap(self, api_key, note_id: str) -> Note:
        """Create SOAP note and return final note to the user"""
        try:
            criteria = {"note_id": note_id}
            note_detail = Note(**self.note_store.load_note(criteria)[0])
        except RecordNotFound as rnf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No note found for note id: {note_id}",
            ) from rnf

        concatenated_raw_note_text = concatenate_transcript_text(transcript_data=note_detail.raw_note)
        soap_note = await create_soap_note(LLM_URL, concatenated_raw_note_text)
        title = extract_title(soap_note)

        note_detail.note_title = title
        note_detail.ai_improved_note = soap_note.rstrip('= \n')
        note = await save_note(self.note_store, note_detail, api_key)
        return note

    async def regenerate_soap(self, note_id: str) -> Note:
        """Regenerate SOAP note based on user action"""
        try:
            criteria = {"note_id": note_id}
            existing_note: Note = Note(**self.note_store.load_note(criteria)[0])
        except RecordNotFound as rnf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note not found with id {note_id}",
            ) from rnf

        diarization_output = existing_note.raw_note
        # Store older note which will be replaced by the latest regenerated note
        if not existing_note.previous_soap_notes:
            existing_note.previous_soap_notes = []
        existing_note.previous_soap_notes.append(existing_note.ai_improved_note)
        concatenated_raw_note_text = concatenate_transcript_text(transcript_data=diarization_output)
        regenerated_soap = await create_soap_note(LLM_URL, concatenated_raw_note_text)
        existing_note.ai_improved_note = regenerated_soap.rstrip('= \n')
        payload_json = jsonable_encoder(existing_note)
        n_rows_modified = self.note_store.update_note(note_id, payload_json)

        if n_rows_modified == 0:
            return {"message": "Could not update note title"}
        else:
            return existing_note.dict()

    async def update(self, note_payload: EditNote) -> Note:
        """Update an existing note and return the modified note."""
        try:
            criteria = {"note_id": note_payload.note_id}
            note_detail = Note(**self.note_store.load_note(criteria)[0])
        except RecordNotFound as rnf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No note found for note id: {note_payload.note_id}",
            ) from rnf
        concatenated_raw_note_text = concatenate_transcript_text(transcript_data=note_payload.modified_note_text)
        if note_payload.note_type == 0:  # Raw note modified, both notes will be updated
            logger.info("Modifying RAW note. Taking it through AI improvement")
            note_detail.raw_note = note_payload.modified_note_text
            note_detail.ai_improved_note = await create_soap_note(
                URL_CHATGPT, concatenated_raw_note_text
            )
        else:
            logger.info("Modifying AI-Improved note. Overriding exising AI note")
            note_detail.ai_improved_note = concatenated_raw_note_text

        note_detail.date_modified = dt.datetime.utcnow()  # modify note update date
        payload_json = jsonable_encoder(note_detail)

        try:
            self.note_store.store_note(payload_json, note_payload.note_id)
        except Exception as exc:
            logging.exception("Something wrong while adding note on routing service.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Something went wrong on routing service.",
            ) from exc
        return payload_json

    def get_all(self, user_id: str) -> List[dict]:
        """Retrieve all historical "notes" from notes collection for the given user ID"""

        criteria = {"user_id": user_id, "is_deleted": False}
        _projection = {
            "date_modified": 1,
            "is_starred": 1,
            "note_title": 1,
            "user_id": 1,
            "note_id": 1,
            "_id": 0,
            "ai_improved_note": 1,
        }
        try:
            return self.note_store.load_note(criteria, projection=_projection)
        except RecordNotFound as rnf:
            logging.info(f"Not notes history found {rnf}")
            return []

    def get_detail(self, note_id: str) -> Note:
        """Retrieve synapti-note details for the given note ID"""

        criteria = {"note_id": note_id}
        try:
            note_detail = self.note_store.load_note(criteria)[0]
        except RecordNotFound as rnf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No note found against note id: {note_id}",
            ) from rnf
        return note_detail

    def update_title(self, note_id: str, request: TitleUpdate) -> Dict[str, str]:
        """Update the title of a specific note."""
        try:
            criteria = {"note_id": note_id, "is_deleted": False}
            existing_note = self.note_store.load_note(criteria)[0]
            existing_note["note_title"] = request.title
            n_rows_modified = self.note_store.update_note(note_id, existing_note)
        except RecordNotFound as rnf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note not found with id {note_id}",
            ) from rnf
        if n_rows_modified == 0:
            return {"message": "Could not update note title"}

        return {"message": f"Note '{note_id}' title updated succesfully"}

    def update_starred_status(self, note_id: str) -> Dict[str, str]:
        """Toggle star note by note_id"""
        try:
            criteria = {"note_id": note_id, "is_deleted": False}
            existing_note = self.note_store.load_note(criteria)[0]
            existing_note["is_starred"] = not existing_note.get("is_starred", False)
            upsert_starred = self.note_store.update_note(note_id, existing_note)
        except RecordNotFound as rnf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note not found against id: {note_id}",
            ) from rnf
        if upsert_starred == 0:
            return {"message": "Could not update note starred status"}

        return {"message": f"Note '{note_id}' starred status updated"}

    def delete(self, note_id: str) -> Dict[str, str]:
        """Delete a note by note ID"""
        try:
            n_rows_modified = self.note_store.delete_note(note_id)
        except RecordNotFound as rnf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note not found against id: {note_id}!",
            ) from rnf
        if n_rows_modified == 0:
            return {"message": f"Could not delete note: {note_id}!"}

        return {"message": f"Note {note_id} succesfully deleted"}

    # pylint: disable=inconsistent-return-statements
    def generate_icd10_codes(self, note_id: str) -> Optional[List[ICD10Code]]:
        """Generating ICD10 codes for a particular note"""
        icd10_codes = {}
        try:
            note = self.note_store.load_note({"note_id": note_id})
        except RecordNotFound as rnf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note missing against note id: {note_id}",
            ) from rnf
        if note:
            note = note[0]
            icd10_codes = get_cached_icd10(note)
            if not icd10_codes:
                all_icd10_codes = get_icd10_codes_aws(note["ai_improved_note"])
                # find unspecified ICD10 child code for every parent code
                icd10_codes = [
                    find_unspecified_icd_child(icd["code"], icd_codes)
                    for icd in all_icd10_codes
                ]
                icd10_codes = filter_unique_codes(icd10_codes)
                icd10_codes = list(islice(icd10_codes, 5))
                note["icd10_codes"] = icd10_codes
                # Update note with ICD10 codes from AWS in db
                self.note_store.store_note(note, note["note_id"])
            return icd10_codes

    def modify_ai_improved_note(self, modified_ai_note_payload: EditImprovedNote) -> dict:
        """Update an existing ai improved note and return the modified note."""
        try:
            criteria = {"note_id": modified_ai_note_payload.note_id}
            note_detail = Note(**self.note_store.load_note(criteria)[0])
        except RecordNotFound as rnf:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No note found for note id: {modified_ai_note_payload.note_id}",
            ) from rnf
        logger.info("Modifying AI-Improved note. Overriding exising AI note")
        note_detail.ai_improved_note = modified_ai_note_payload.modified_ai_note_text
        # modify note update date
        note_detail.date_modified = dt.datetime.utcnow()
        payload_json = jsonable_encoder(note_detail)
        try:
            self.note_store.store_note(payload_json, modified_ai_note_payload.note_id)
        except Exception as exc:
            logging.exception("Something wrong while adding note on routing service.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Something went wrong on routing service.",
            ) from exc
        return payload_json