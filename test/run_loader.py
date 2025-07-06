# run_loader.py
from logic.utils import load_exhibition_data, load_artifact_data, load_accessibility_data, load_educational_programs_data, load_location_data, load_museum_collection_data, load_research_data, load_safety_info_data, load_special_features_data, load_visitor_reviews_data, load_visitor_services_data, load_all_docs                                            
from pprint import pprint

# Load exhibition data
exhibition_docs = load_exhibition_data(r"D:\Job\balzi-rossi-bot\data\exhibittion.json")
print(f"Loaded {len(exhibition_docs)} exhibition documents.")
pprint(exhibition_docs[0].model_dump())  # Preview first one

print("\n" + "="*80 + "\n")

# Load artifact data
artifact_docs = load_artifact_data("data/artifacts.json")
print(f"Loaded {len(artifact_docs)} artifact documents.")
pprint(artifact_docs[0].model_dump())  # Preview first one

print("\n" + "="*80 + "\n")

access_docs = load_accessibility_data(r"D:\Job\balzi-rossi-bot\data\accessability.json")
print(f"Loaded {len(access_docs)} accessibility documents.")
if access_docs:
    pprint(access_docs[0].model_dump())

print("\n" + "="*80 + "\n")

edu_docs = load_educational_programs_data("data/educational_programs.json")
print(f"Loaded {len(edu_docs)} educational program documents.")
if edu_docs:
    pprint(edu_docs[0].model_dump())

print("\n" + "="*80 + "\n")

location_docs = load_location_data("data/location_direction.json")
print(f"Loaded {len(location_docs)} location documents.")
if location_docs:
    pprint(location_docs[0].model_dump())

print("\n" + "="*80 + "\n")

museum_docs = load_museum_collection_data("data/museum_collection.json")
print(f"Loaded {len(museum_docs)} museum collection documents.")
if museum_docs:
    pprint(museum_docs[0].model_dump())

print("\n" + "="*80 + "\n")

research_docs = load_research_data("data/research.json")
print(f"Loaded {len(research_docs)} research documents.")
if research_docs:
    pprint(research_docs[0].model_dump())

print("\n" + "="*80 + "\n")

docs = load_safety_info_data("data/safety_info.json")
print(f"Loaded {len(docs)} safety documents.")
pprint(docs[0].model_dump())

print("\n" + "="*80 + "\n")

docs = load_special_features_data(r"D:\Job\balzi-rossi-bot\data\special_features.json")
print(f"Loaded {len(docs)} special feature documents.")
pprint(docs[0].model_dump())

print("\n" + "="*80 + "\n")

docs = load_visitor_reviews_data("data/visitor_reviews.json")
print(f"Loaded {len(docs)} visitor reviews documents.")
pprint(docs[0].model_dump())

print("\n" + "="*80 + "\n")

docs = load_visitor_services_data("data/visitor_servise.json")
print(f"Loaded {len(docs)} visitor reviews documents.")
pprint(docs[0].model_dump())

print("\n" + "="*80 + "\n")

def test_load_all_docs():
    print("🔍 Loading all documents...\n")
    docs = load_all_docs(base_path="data")  # You can change this path if needed

    print(f"\n📚 Total documents loaded: {len(docs)}\n")
    
    if docs:
        print("📌 Sample document:\n")
        pprint(docs[0].model_dump())

if __name__ == "__main__":
    test_load_all_docs()
    