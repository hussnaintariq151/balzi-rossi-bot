import json
import hashlib
from typing import Any, Dict, List
from langchain_core.documents import Document


DEBUG_MODE = False  # Set to True during development


def _load_json_file(file_path: str) -> Dict[str, Any]:
    """Safely loads JSON from a file. Returns empty dict if invalid."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[Warning] File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"[Warning] Failed to parse JSON in: {file_path}")
    except Exception as e:
        print(f"[Warning] Unexpected error reading {file_path}: {e}")
        if DEBUG_MODE:
            raise
    return {}


def _get_multilingual_field(item: dict, field: str, lang: str) -> str | list:
    """Returns field[lang] if exists, else empty string or list."""
    value = item.get(field, {})
    if isinstance(value, dict):
        return value.get(lang, "")
    elif isinstance(value, list):
        return [entry.get(lang, "") for entry in value]
    return ""


def _generate_id(obj: dict, prefix: str = "doc") -> str:
    """Generates a stable unique ID for an object."""
    obj_str = json.dumps(obj, sort_keys=True)
    return f"{prefix}_{hashlib.md5(obj_str.encode()).hexdigest()}"

def load_exhibition_data(file_path: str) -> List[Document]:
    """
    Loads and processes multilingual exhibition and event data from JSON.
    Returns a list of LangChain Document objects with text and metadata.
    """
    data = _load_json_file(file_path)
    if not data:
        return []

    documents: List[Document] = []
    museum_name = data.get("museum", "Balzi Rossi Prehistoric Museum")
    languages = ["en", "it", "fr", "de", "ar"]

    # --- Process Permanent Exhibitions ---
    for exhibit in data.get("exhibitions", []):
        base_id = exhibit.get("id", _generate_id(exhibit, prefix="exhibit"))

        for lang in languages:
            title = _get_multilingual_field(exhibit, "title", lang)
            exhibit_type = _get_multilingual_field(exhibit, "type", lang)
            description = _get_multilingual_field(exhibit, "description", lang)
            features = _get_multilingual_field(exhibit, "features", lang)

            content_parts = []
            if title: content_parts.append(f"Title: {title}")
            if exhibit_type: content_parts.append(f"Type: {exhibit_type}")
            if description: content_parts.append(f"Description: {description}")
            if features: content_parts.append(f"Features: {', '.join(features)}")

            text_content = ". ".join(content_parts) + "." if content_parts else ""

            metadata = {
                "doc_id": base_id,
                "doc_type": "exhibition",
                "language": lang,
                "museum": museum_name,
                "title_en": exhibit.get("title", {}).get("en"),
                "type_en": exhibit.get("type", {}).get("en"),
                "start_date": exhibit.get("start_date"),
                "end_date": exhibit.get("end_date"),
                "source": exhibit.get("source"),
                "length": len(text_content.split()),
            }

            documents.append(Document(page_content=text_content, metadata=metadata))

    # --- Process Special Events ---
    for event in data.get("special_events", []):
        base_id = event.get("id", _generate_id(event, prefix="event"))

        for lang in languages:
            title = _get_multilingual_field(event, "title", lang)
            event_type = _get_multilingual_field(event, "type", lang)
            description = _get_multilingual_field(event, "description", lang)
            highlights = _get_multilingual_field(event, "highlights", lang)

            content_parts = []
            if title: content_parts.append(f"Title: {title}")
            if event_type: content_parts.append(f"Type: {event_type}")
            if description: content_parts.append(f"Description: {description}")
            if highlights: content_parts.append(f"Highlights: {', '.join(highlights)}")

            text_content = ". ".join(content_parts) + "." if content_parts else ""

            metadata = {
                "doc_id": base_id,
                "doc_type": "special_event",
                "language": lang,
                "museum": museum_name,
                "title_en": event.get("title", {}).get("en"),
                "type_en": event.get("type", {}).get("en"),
                "start_date": event.get("start_date"),
                "end_date": event.get("end_date"),
                "source": event.get("source"),
                "length": len(text_content.split()),
            }

            documents.append(Document(page_content=text_content, metadata=metadata))

    return documents

def _generate_id(obj: dict, prefix: str = "doc") -> str:
    """Generates a stable, hash-based ID from a JSON-serializable object.
    Ensures consistent IDs across runs and environments.
    """
    # Sort keys to ensure consistent JSON string representation for hashing
    obj_str = json.dumps(obj, sort_keys=True, ensure_ascii=False) # Added ensure_ascii=False for non-ASCII characters
    return f"{prefix}_{hashlib.md5(obj_str.encode('utf-8')).hexdigest()}" # Explicitly encode to utf-8


def load_artifact_data(file_path: str) -> List[Document]:
    """
    Loads and processes multilingual artifact data from artifact.json.
    Creates LangChain Document objects for both main artifacts and their sub-artifacts,
    including detailed text content and rich metadata for each language.
    """
    data = _load_json_file(file_path)
    if not data:
        return []

    documents: List[Document] = []
    languages = ["en", "it", "fr", "de", "ar"]

    for item in data:
        main_artifact_id = item.get("id", _generate_id(item, prefix="main_artifact"))

        for lang in languages:
            # --- Main Artifact/Burial Document ---
            name = _get_multilingual_field(item, "name", lang)
            item_type = item.get("type")
            period = item.get("period")
            estimated_date = item.get("estimated_date")
            origin = _get_multilingual_field(item, "origin", lang)
            description = _get_multilingual_field(item, "description", lang)
            significance = _get_multilingual_field(item, "significance", lang)
            material = _get_multilingual_field(item, "material", lang)

            content_parts = []
            if name: content_parts.append(f"Name: {name}")
            if item_type: content_parts.append(f"Type: {item_type}")
            if period: content_parts.append(f"Period: {period}")
            if estimated_date: content_parts.append(f"Estimated Date: {estimated_date}")
            if origin: content_parts.append(f"Origin: {origin}")
            if material:
                if isinstance(material, list):
                    content_parts.append(f"Material: {', '.join(material)}")
                else:
                    content_parts.append(f"Material: {material}")
            if description: content_parts.append(f"Description: {description}")
            if significance: content_parts.append(f"Significance: {significance}")

            text_content = ". ".join(filter(None, content_parts)).strip()
            if text_content:
                text_content += "."

            metadata = {
                "doc_id": main_artifact_id,
                "doc_type": item_type.lower().replace(" ", "_") if item_type else "artifact",
                "language": lang,
                "name_en": item.get("name", {}).get("en"),
                "type_en": item.get("type") if isinstance(item.get("type"), str) else "",
                "period_en": item.get("period") if isinstance(item.get("period"), str) else "",
                "estimated_date": estimated_date,
                "accession_number": item.get("accession_number"),
                "origin_en": item.get("origin", {}).get("en") if isinstance(item.get("origin"), dict) else "",
                "material_en": item.get("material", {}).get("en") if isinstance(item.get("material"), dict) else "",
                "curator": item.get("curator"),
                "source": item.get("source"),
                "length": len(text_content.split())
            }

            documents.append(Document(page_content=text_content, metadata=metadata))

            # --- Sub-Artifacts ---
            for sub_artifact in item.get("sub_artifacts", []):
                sub_artifact_id = sub_artifact.get("id", _generate_id(sub_artifact, prefix=f"{main_artifact_id}_sub"))

                sub_name = _get_multilingual_field(sub_artifact, "name", lang)
                sub_material = sub_artifact.get("material")
                sub_description = _get_multilingual_field(sub_artifact, "description", lang)
                sub_estimated_date = sub_artifact.get("estimated_date")

                sub_content_parts = []
                if sub_name: sub_content_parts.append(f"Sub-artifact Name: {sub_name}")
                if name: sub_content_parts.append(f"(Part of: {name})")
                if sub_material: sub_content_parts.append(f"Material: {sub_material}")
                if sub_estimated_date: sub_content_parts.append(f"Estimated Date: {sub_estimated_date}")
                if sub_description: sub_content_parts.append(f"Description: {sub_description}")

                sub_text_content = ". ".join(filter(None, sub_content_parts)).strip()
                if sub_text_content:
                    sub_text_content += "."

                sub_metadata = {
                    "doc_id": sub_artifact_id,
                    "doc_type": "sub_artifact",
                    "language": lang,
                    "parent_artifact_id": main_artifact_id,
                    "parent_artifact_name_en": item.get("name", {}).get("en"),
                    "name_en": sub_artifact.get("name", {}).get("en") if isinstance(sub_artifact.get("name"), dict) else "",
                    "material": sub_material,
                    "estimated_date": sub_estimated_date,
                    "accession_number": sub_artifact.get("accession_number"),
                    "curator": sub_artifact.get("curator"),
                    "length": len(sub_text_content.split())
                }

                documents.append(Document(page_content=sub_text_content, metadata=sub_metadata))

    return documents


def load_accessibility_data(file_path: str) -> List[Document]:
    """
    Loads multilingual accessibility information from accessibility.json.
    Returns one Document per language with detailed features and notes.
    """
    data = _load_json_file(file_path)
    if not data:
        return []

    documents: List[Document] = []
    languages = ["en", "it", "fr", "de", "ar"]

    access_data = data.get("accessibility", {})
    base_id = data.get("id", _generate_id(access_data, prefix="accessibility"))

    for lang in languages:
        title = _get_multilingual_field(access_data, "title", lang)
        description = _get_multilingual_field(access_data, "description", lang)
        additional_notes = _get_multilingual_field(access_data, "additional_notes", lang)

        # Extract features list
        features = []
        for f in access_data.get("features", []):
            feature_text = f.get("feature", {}).get(lang, "")
            if feature_text:
                features.append(feature_text)

        # Combine into one readable text content
        content_parts = []
        if title: content_parts.append(f"Title: {title}")
        if description: content_parts.append(f"Description: {description}")
        if features:
            content_parts.append("Features: " + "; ".join(features))
        if additional_notes: content_parts.append(f"Additional Notes: {additional_notes}")

        text_content = ". ".join(filter(None, content_parts)).strip()
        if text_content:
            text_content += "."

        metadata = {
            "doc_id": f"{base_id}_{lang}",
            "doc_type": "accessibility",
            "language": lang,
            "title_en": access_data.get("title", {}).get("en"),
            "length": len(text_content.split())
        }

        documents.append(Document(page_content=text_content, metadata=metadata))

    return documents


def load_educational_programs_data(file_path: str) -> List[Document]:
    """
    Loads and processes multilingual educational programs from educational_programs.json.
    Creates one Document per program section (school, university, public, calendar) per language.
    """
    data = _load_json_file(file_path)
    if not data:
        return []

    programs = data.get("educational_programs", {})
    base_id = data.get("id", _generate_id(programs, prefix="edu_programs"))

    languages = ["en", "it", "fr", "de", "ar"]
    documents: List[Document] = []

    # --- School Programs ---
    school = programs.get("school_programs", {})
    for lang in languages:
        description = _get_multilingual_field(school, "description", lang)
        features = [
            f"Age Groups: {', '.join(school.get('age_groups', []))}" if school.get("age_groups") else "",
            f"Languages: {', '.join(school.get('languages', []))}" if school.get("languages") else "",
            f"Cost: {school.get('cost', '')}",
        ]
        booking_info = school.get("booking_info", {})
        contact = booking_info.get("contact")
        phone = booking_info.get("phone")

        content_parts = [
            f"Section: School Programs",
            f"Description: {description}" if description else "",
            *features,
            f"Booking Contact: {contact}" if contact else "",
            f"Phone: {phone}" if phone else ""
        ]

        text_content = ". ".join(filter(None, content_parts)).strip()
        if text_content:
            text_content += "."

        metadata = {
            "doc_id": f"{base_id}_school_{lang}",
            "doc_type": "educational_school",
            "language": lang,
            "section": "school_programs",
            "length": len(text_content.split()),
        }

        documents.append(Document(page_content=text_content, metadata=metadata))

    # --- University Collaborations ---
    uni = programs.get("university_collaborations", {})
    for lang in languages:
        description = _get_multilingual_field(uni, "description", lang)
        universities = uni.get("universities", [])

        uni_details = []
        for u in universities:
            parts = [u.get("name", "")]
            if "departments" in u:
                parts.append("Departments: " + ", ".join(u["departments"]))
            if "role" in u:
                parts.append("Role: " + u["role"])
            uni_details.append("; ".join(filter(None, parts)))

        content_parts = [
            f"Section: University Collaborations",
            f"Description: {description}" if description else "",
            "Partners: " + " | ".join(uni_details) if uni_details else ""
        ]

        text_content = ". ".join(filter(None, content_parts)).strip()
        if text_content:
            text_content += "."

        metadata = {
            "doc_id": f"{base_id}_university_{lang}",
            "doc_type": "educational_university",
            "language": lang,
            "section": "university_collaborations",
            "length": len(text_content.split()),
        }

        documents.append(Document(page_content=text_content, metadata=metadata))

    # --- Public Workshops ---
    public = programs.get("public_workshops", {})
    for lang in languages:
        description = _get_multilingual_field(public, "description", lang)

        content_parts = [
            f"Section: Public Workshops",
            f"Description: {description}" if description else "",
        ]

        text_content = ". ".join(filter(None, content_parts)).strip()
        if text_content:
            text_content += "."

        metadata = {
            "doc_id": f"{base_id}_public_{lang}",
            "doc_type": "educational_public",
            "language": lang,
            "section": "public_workshops",
            "length": len(text_content.split()),
        }

        documents.append(Document(page_content=text_content, metadata=metadata))

    # --- Events Calendar ---
    calendar = programs.get("events_calendar", {})
    for lang in languages:
        note = _get_multilingual_field(calendar, "note", lang)

        content_parts = [
            f"Section: Events Calendar",
            f"Note: {note}" if note else "",
        ]

        text_content = ". ".join(filter(None, content_parts)).strip()
        if text_content:
            text_content += "."

        metadata = {
            "doc_id": f"{base_id}_calendar_{lang}",
            "doc_type": "educational_calendar",
            "language": lang,
            "section": "events_calendar",
            "length": len(text_content.split()),
        }

        documents.append(Document(page_content=text_content, metadata=metadata))

    return documents


def load_location_data(file_path: str) -> List[Document]:
    """
    Loads and processes multilingual location & directions data from location_direction.json.
    Returns one Document per language with directions, address, parking info, and landmarks.
    """
    data = _load_json_file(file_path)
    if not data:
        return []

    documents: List[Document] = []
    base_id = data.get("id", _generate_id(data, prefix="location"))
    languages = ["en", "it", "fr", "de", "ar"]

    location = data.get("location", {})
    how_to = data.get("how_to_get_there", {})
    landmarks = data.get("nearby_landmarks", {})
    parking = data.get("parking_info", {})

    latitude = location.get("latitude")
    longitude = location.get("longitude")

    for lang in languages:
        address = _get_multilingual_field(location, "address", lang)
        by_train = _get_multilingual_field(how_to, "by_train", lang)
        by_car = _get_multilingual_field(how_to, "by_car", lang)
        walking_path = _get_multilingual_field(how_to, "walking_path", lang)
        landmark_list = landmarks.get(lang, [])
        parking_info = parking.get(lang, "")

        content_parts = [
            f"Address: {address}" if address else "",
            f"By Train: {by_train}" if by_train else "",
            f"By Car: {by_car}" if by_car else "",
            f"Walking Path: {walking_path}" if walking_path else "",
            f"Nearby Landmarks: {', '.join(landmark_list)}" if landmark_list else "",
            f"Parking Info: {parking_info}" if parking_info else "",
            f"Coordinates: Latitude {latitude}, Longitude {longitude}" if latitude and longitude else ""
        ]

        text_content = ". ".join(filter(None, content_parts)).strip()
        if text_content:
            text_content += "."

        metadata = {
            "doc_id": f"{base_id}_{lang}",
            "doc_type": "location",
            "language": lang,
            "address_en": location.get("address", {}).get("en"),
            "latitude": latitude,
            "longitude": longitude,
            "length": len(text_content.split())
        }

        documents.append(Document(page_content=text_content, metadata=metadata))

    return documents

def load_museum_collection_data(file_path: str) -> List[Document]:
    """
    Loads and processes multilingual museum collection data from museum_collection.json.
    Creates one LangChain Document per language combining overview, artifacts, and exhibits.
    """
    data = _load_json_file(file_path)
    if not data:
        return []

    documents: List[Document] = []
    base_id = data.get("id", _generate_id(data, prefix="museum_collection"))
    collections = data.get("collections", {})
    languages = ["en", "it", "fr", "de", "ar"]

    for lang in languages:
        overview = _get_multilingual_field(collections, "overview", lang)
        exhibits = _get_multilingual_field(collections, "exhibits", lang)

        # Process notable artifacts list
        artifacts = collections.get("notable_artifacts", [])
        artifact_texts = []
        for artifact in artifacts:
            name = _get_multilingual_field(artifact, "name", lang)
            description = _get_multilingual_field(artifact, "description", lang)
            if name or description:
                artifact_texts.append(f"{name}: {description}")

        content_parts = [
            f"Section: Museum Collection Overview",
            f"Overview: {overview}" if overview else "",
            f"Notable Artifacts: {' | '.join(artifact_texts)}" if artifact_texts else "",
            f"Exhibits: {exhibits}" if exhibits else ""
        ]

        text_content = ". ".join(filter(None, content_parts)).strip()
        if text_content:
            text_content += "."

        metadata = {
            "doc_id": f"{base_id}_{lang}",
            "doc_type": "museum_collection",
            "language": lang,
            "overview_en": collections.get("overview", {}).get("en"),
            "length": len(text_content.split())
        }

        documents.append(Document(page_content=text_content, metadata=metadata))

    return documents



def load_research_data(file_path: str) -> List[Document]:
    """
    Loads and processes multilingual research data from research.json.
    Returns one Document per language covering research projects, excavations, and outreach.
    """
    data = _load_json_file(file_path)
    if not data:
        return []

    documents: List[Document] = []
    base_id = data.get("id", _generate_id(data, prefix="research"))
    research = data.get("research", {})
    languages = ["en", "it", "fr", "de", "ar"]

    ongoing_projects = research.get("ongoing_projects", [])
    recent_excavations = research.get("recent_excavations", {})
    public_engagement = research.get("public_engagement", {})

    for lang in languages:
        content_parts = []

        # --- Ongoing Projects ---
        if ongoing_projects:
            for project in ongoing_projects:
                name = project.get("name")
                description = _get_multilingual_field(project, "description", lang)
                universities = project.get("collaborating_universities", [])
                funding = project.get("funding", {})
                funding_org = funding.get("organization")
                funding_amount = funding.get("amount")
                program = funding.get("program")
                start_year = project.get("start_year")
                status = project.get("status")

                project_parts = [
                    f"Project: {name}" if name else "",
                    f"Description: {description}" if description else "",
                    f"Status: {status}" if status else "",
                    f"Start Year: {start_year}" if start_year else "",
                    f"Collaborating Universities: {', '.join(universities)}" if universities else "",
                    f"Funding: {funding_org} - {funding_amount} ({program})" if funding else ""
                ]
                content_parts.append(". ".join(filter(None, project_parts)))

        # --- Recent Excavations ---
        excavation_available = recent_excavations.get("available", False)
        if excavation_available:
            excavation_desc = _get_multilingual_field(recent_excavations, "description", lang)
            if excavation_desc:
                content_parts.append(f"Recent Excavations: {excavation_desc}")

        # --- Public Engagement ---
        engagement_available = public_engagement.get("available", False)
        if engagement_available:
            engagement_desc = _get_multilingual_field(public_engagement, "description", lang)
            if engagement_desc:
                content_parts.append(f"Public Engagement: {engagement_desc}")

        text_content = ". ".join(filter(None, content_parts)).strip()
        if text_content:
            text_content += "."

        metadata = {
            "doc_id": f"{base_id}_{lang}",
            "doc_type": "research",
            "language": lang,
            "main_project_en": ongoing_projects[0].get("name") if ongoing_projects else "",
            "start_year": ongoing_projects[0].get("start_year") if ongoing_projects else "",
            "status": ongoing_projects[0].get("status") if ongoing_projects else "",
            "length": len(text_content.split())
        }

        documents.append(Document(page_content=text_content, metadata=metadata))

    return documents


def load_safety_info_data(file_path: str) -> List[Document]:
    """
    Loads multilingual visitor safety data from safety_info.json.
    Creates one Document per language, including safety description, guidelines, and emergency contacts.
    """
    data = _load_json_file(file_path)
    if not data:
        return []

    documents: List[Document] = []
    base_id = data.get("id", _generate_id(data, prefix="safety"))
    safety_info = data.get("safety_information", {})
    languages = ["en", "it", "fr", "de", "ar"]

    guidelines = safety_info.get("guidelines", [])
    emergency_contacts = safety_info.get("emergency", {}).get("contacts", [])

    for lang in languages:
        content_parts = []

        # Title & Description
        title = _get_multilingual_field(safety_info, "title", lang)
        description = _get_multilingual_field(safety_info, "description", lang)

        if title:
            content_parts.append(f"Title: {title}")
        if description:
            content_parts.append(f"Description: {description}")

        # Guidelines
        if guidelines:
            bullet_points = []
            for entry in guidelines:
                point = _get_multilingual_field(entry, "point", lang)
                if point:
                    bullet_points.append(f"- {point}")
            if bullet_points:
                content_parts.append("Guidelines:\n" + "\n".join(bullet_points))

        # Emergency Contacts
        if emergency_contacts:
            contact_lines = []
            for contact in emergency_contacts:
                contact_lines.append(f"{contact['name']}: {contact['phone']}")
            content_parts.append("Emergency Contacts:\n" + "\n".join(contact_lines))

        text_content = "\n\n".join(content_parts).strip()

        metadata = {
            "doc_id": f"{base_id}_{lang}",
            "doc_type": "safety_info",
            "language": lang,
            "length": len(text_content.split())
        }

        documents.append(Document(page_content=text_content, metadata=metadata))

    return documents


def load_special_features_data(file_path: str) -> List[Document]:
    """
    Loads multilingual special feature data from special_features.json.
    Each document includes features like landscape, caves, sea views, etc. — one per language.
    """
    data = _load_json_file(file_path)
    if not data:
        return []

    documents: List[Document] = []
    special_data = data.get("special_features", {})
    base_id = data.get("id", _generate_id(data, prefix="special_features"))
    languages = ["en", "it", "fr", "de", "ar"]

    for lang in languages:
        content_parts = []

        for feature_key, translations in special_data.items():
            if not isinstance(translations, dict):
                continue  # Skip malformed fields

            # Get the translated description
            description = translations.get(lang, "").strip()
            if not description:
                continue

            # Format title from key
            title = feature_key.replace("_", " ").title().replace(" And ", " and ")
            content_parts.append(f"{title}: {description}")

        # Join all content for this language
        text_content = "\n\n".join(content_parts).strip()
        if not text_content:
            continue

        metadata = {
            "doc_id": f"{base_id}_{lang}",
            "doc_type": "special_feature",
            "language": lang,
            "length": len(text_content.split())
        }

        documents.append(Document(page_content=text_content, metadata=metadata))

    return documents


def load_visitor_reviews_data(file_path: str) -> List[Document]:
    """
    Loads multilingual visitor reviews and feedback from visitor_reviews.json.
    Generates documents for summaries, highlights, and example reviews.
    """
    data = _load_json_file(file_path)
    if not data:
        return []

    documents: List[Document] = []
    base_id = data.get("id", _generate_id(data, prefix="visitor_reviews")).rstrip("_")
    reviews = data.get("visitor_reviews_and_feedback", {})
    languages = ["en", "it", "fr", "de", "ar"]

    # 1. Summary documents (one per language)
    summary_data = reviews.get("summary", {})
    for lang in languages:
        summary = summary_data.get(lang, "").strip()
        if not summary:
            continue

        documents.append(Document(
            page_content=summary,
            metadata={
                "doc_id": f"{base_id}_summary_{lang}",
                "doc_type": "visitor_summary",
                "language": lang,
                "avg_rating": reviews.get("average_rating")
            }
        ))

    # 2. Highlight documents (per aspect per language)
    for i, highlight in enumerate(reviews.get("highlights", [])):
        aspects = highlight.get("aspect", {})
        comments = highlight.get("comment", {})
        for lang in languages:
            aspect_text = aspects.get(lang, "").strip()
            comment_text = comments.get(lang, "").strip()
            if not aspect_text or not comment_text:
                continue

            documents.append(Document(
                page_content=f"{aspect_text}: {comment_text}",
                metadata={
                    "doc_id": f"{base_id}_highlight_{i}_{lang}",
                    "doc_type": "highlight",
                    "language": lang,
                    "aspect_en": aspects.get("en", "")
                }
            ))

    # 3. Example review documents
    for i, review in enumerate(reviews.get("example_reviews", [])):
        text = review.get("text", "").strip()
        if not text:
            continue

        documents.append(Document(
            page_content=text,
            metadata={
                "doc_id": f"{base_id}_review_{i}_{review.get('language', 'unknown')}",
                "doc_type": "example_review",
                "language": review.get("language", "unknown"),
                "rating": review.get("rating")
            }
        ))

    return documents

def load_visitor_services_data(file_path: str) -> List[Document]:
    """
    Loads multilingual visitor services data from visitor_services.json.
    Generates documents summarizing amenities, accessibility, family-friendliness, tours, and general description.
    """
    data = _load_json_file(file_path)
    if not data:
        return []

    documents: List[Document] = []
    base_id = data.get("id", _generate_id(data, prefix="visitor_services")).rstrip("_")
    services = data.get("visitor_services", {})
    languages = ["en", "it", "fr", "de", "ar"]

    # Compose shared service info summary (non-translated parts)
    static_features = {
        "Accessibility": services.get("accessibility", {}),
        "Amenities": services.get("amenities", {}),
        "Rest Areas": services.get("rest_areas", {}),
        "Family Friendly": services.get("family_friendly", {}),
        "Guided Tours": {
            k: v for k, v in services.get("guided_tours", {}).items()
            if k not in ("description",)  # exclude multilingual description
        }
    }

    static_summary_lines = []
    for section, features in static_features.items():
        for key, value in features.items():
            if isinstance(value, bool):
                line = f"{section} – {key.replace('_', ' ').title()}: {'Yes' if value else 'No'}"
                static_summary_lines.append(line)

    static_summary = "\n".join(static_summary_lines).strip()

    # Build document per language
    for lang in languages:
        lang_parts = []

        # Add multilingual description
        general_description = services.get("description", {}).get(lang, "").strip()
        if general_description:
            lang_parts.append(f"General Description:\n{general_description}")

        # Add guided tours description if available
        guided_tours_desc = services.get("guided_tours", {}).get("description", {}).get(lang, "").strip()
        if guided_tours_desc:
            lang_parts.append(f"Guided Tours:\n{guided_tours_desc}")

        # Add static features
        if static_summary:
            lang_parts.append(f"Service Highlights:\n{static_summary}")

        text_content = "\n\n".join(lang_parts).strip()
        if not text_content:
            continue

        metadata = {
            "doc_id": f"{base_id}_{lang}",
            "doc_type": "visitor_services",
            "language": lang,
            "length": len(text_content.split())
        }

        documents.append(Document(page_content=text_content, metadata=metadata))

    return documents

import logging
import os


def load_all_docs(selected_sources=None, base_path="data") -> List[Document]:
    source_map = {
        "exhibition": ("exhibittion.json", load_exhibition_data),
        "artifact": ("artifacts.json", load_artifact_data),
        "accessibility": ("accessability.json", load_accessibility_data),
        "education": ("educational_programs.json", load_educational_programs_data),
        "location": ("location_direction.json", load_location_data),
        "collection": ("museum_collection.json", load_museum_collection_data),
        "research": ("research.json", load_research_data),
        "safety": ("safety_info.json", load_safety_info_data),
        "features": ("special_features.json", load_special_features_data),
        "reviews": ("visitor_reviews.json", load_visitor_reviews_data),
        "services": ("visitor_servise.json", load_visitor_services_data),
    }

    selected_sources = selected_sources or list(source_map.keys())
    all_docs = []

    for source in selected_sources:
        if source in source_map:
            file_name, loader_fn = source_map[source]
            file_path = os.path.join(base_path, file_name)
            try:
                docs = loader_fn(file_path)
                logging.info(f"✅ Loaded {len(docs)} docs from `{source}`")
                all_docs.extend(docs)
            except Exception as e:
                logging.error(f"❌ Failed to load `{source}`: {e}")
        else:
            logging.warning(f"⚠️ Unknown source key: {source}")

    return all_docs
