import json
import os
import random
from typing import List, Dict, Any

def extract_text_from_field(field_value):
    """Extract text from field that could be string, list, or dict."""
    if isinstance(field_value, str):
        return field_value
    elif isinstance(field_value, list):
        return ' '.join(str(item) for item in field_value if item)
    elif isinstance(field_value, dict):
        for key in ['text', 'content', 'abstract', 'title']:
            if key in field_value:
                return str(field_value[key])
        return ' '.join(str(v) for v in field_value.values() if v)
    else:
        return str(field_value) if field_value else ''

def load_jsonl_file(filepath: str) -> List[Dict]:
    """Load a JSONL file safely."""
    data = []
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return data
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {filepath}: {e}")
        print(f"Loaded {len(data)} entries from {os.path.basename(filepath)}")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return data

def load_json_file(filepath: str) -> Dict:
    """Load a JSON file safely."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")cs
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded JSON from {os.path.basename(filepath)}")
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

def create_search_results(claim: Dict[str, Any], corpus_data: List[Dict]) -> List[Dict[str, Any]]:
    """Create search results for a claim using relevant doc_ids."""
    search_results = []
    
    # Get doc_ids from the claim
    doc_ids = claim.get('doc_ids', [])
    
    # Create a mapping from doc_id to corpus document
    corpus_dict = {str(doc['doc_id']): doc for doc in corpus_data}  # Use doc_id as key
    
    # Get up to 5 documents based on doc_ids
    for i, doc_id in enumerate(doc_ids[:5]):  # Take first 5 doc_ids
        doc_id_str = str(doc_id)  # Convert to string for consistency
        if doc_id_str in corpus_dict:
            doc = corpus_dict[doc_id_str]
            # page_name = title from corpus
            page_name = doc.get('title', f'Health Document {doc_id}')
            # page_result = abstract from corpus
            page_result = doc.get('abstract', ['Health-related content'])
            if isinstance(page_result, list):
                page_result = ' '.join(page_result)  # Join sentences into a single string
            else:
                page_result = str(page_result) if page_result else 'Health-related content'
        else:
            # Fallback if doc_id not found in corpus
            page_name = f'Health Document {doc_id}'
            page_result = 'Health-related content'
        
        search_result = {
            "page_name": page_name,
            "page_url": "",  # Empty as requested
            "page_snippet": "",  # Empty as requested (rỗng)
            "page_result": page_result,
            "page_last_modified": ""
        }
        search_results.append(search_result)
    
    # If we have less than 5 doc_ids, fill with random docs to make 5 total
    while len(search_results) < 5:
        if corpus_data:
            doc = random.choice(corpus_data)
            # For random docs, use available title/abstract
            page_name = doc.get('title', f'Health Document {len(search_results)+1}')
            page_result = doc.get('abstract', ['Health-related content'])
            if isinstance(page_result, list):
                page_result = ' '.join(page_result)
            else:
                page_result = str(page_result) if page_result else 'Health-related content'
        else:
            page_name = f'Health Document {len(search_results)+1}'
            page_result = 'Health-related content'
        
        search_result = {
            "page_name": page_name,
            "page_url": "",
            "page_snippet": "",  # Empty (rỗng)
            "page_result": page_result,
            "page_last_modified": ""
        }
        search_results.append(search_result)
    
    return search_results

def convert_healthver_to_mcqa(healthver_dir: str, output_file: str):
    """Convert HealthVer dataset to MCQA format."""
    
    print(f"Looking for HealthVer files in: {healthver_dir}")
    
    # Load all claims files
    all_claims = []
    claim_files = {
        'claims_dev.jsonl': 'dev',
        'claims_test.jsonl': 'test',
        'claims_train.jsonl': 'train',
        'claims_fewshot.jsonl': 'fewshot'
    }
    
    print("\n=== Loading Claims Files ===")
    for filename, split_name in claim_files.items():
        filepath = os.path.join(healthver_dir, filename)
        claims_data = load_jsonl_file(filepath)
        
        # Add source split info
        for claim in claims_data:
            claim['source_split'] = split_name
        
        all_claims.extend(claims_data)
    
    print(f"\nTotal claims loaded: {len(all_claims)}")
    
    # Debug: Print first claim structure
    if all_claims:
        print("\n=== Sample Claim Structure ===")
        print(json.dumps(all_claims[0], indent=2, ensure_ascii=False))
    
    if len(all_claims) == 0:
        print("No claims found! Please check:")
        print("1. Directory path is correct")
        print("2. Files exist: claims_dev.jsonl, claims_test.jsonl, claims_train.jsonl, claims_fewshot.jsonl")
        return
    
    # Load corpus (optional)
    corpus_path = os.path.join(healthver_dir, 'corpus.jsonl')
    corpus_data = load_jsonl_file(corpus_path)
    
    # Debug: Print first corpus entry structure
    if corpus_data:
        print("\n=== Sample Corpus Structure ===")
        print(json.dumps(corpus_data[0], indent=2, ensure_ascii=False))
    
    # Shuffle claims
    random.shuffle(all_claims)
    
    # Create MCQA structure
    total_samples = len(all_claims)
    calibration_samples = min(50, max(5, total_samples // 4))  # 25% or max 50, min 5
    
    mcqa_data = {
        "name": "HEALTHVER_MCQA",
        "description": "HealthVer dataset converted to MCQA format for evaluating health claim verification systems.",
        "version": "1.0",
        "total_samples": total_samples,
        "calibration_samples": calibration_samples,
        "test_samples": total_samples - calibration_samples,
        "calibration": [],
        "test": []  # Added test array
    }
    
    print(f"\n=== Creating MCQA Dataset ===")
    print(f"Total samples: {total_samples}")
    print(f"Calibration samples: {calibration_samples}")
    print(f"Test samples: {total_samples - calibration_samples}")
    
    # Process claims
    for i, claim in enumerate(all_claims):
        if i % 1000 == 0:
            print(f"Processing claim {i+1}/{total_samples}")
        
        claim_text = claim.get('claim', 'No claim text')
        label = claim.get('label', 'NEI')
        
        # Map labels to answers
        if label.upper() == 'SUPPORTS':
            correct_answer = "A"
        elif label.upper() == 'REFUTES':
            correct_answer = "B"
        elif label.upper() == 'NEI':
            correct_answer = "C"
        
        # Create question
        question = f"Is the following health claim supported by evidence: {claim_text}\nA. Supported\nB. Refuted\nC. Not Enough Information"
        
        # Create search results
        search_results = create_search_results(claim, corpus_data)
        
        # Create sample
        sample = {
            "id": claim.get('id', i),
            "question": question,
            "correct_answer": correct_answer,
            "options": ["A", "B", "C"],
            "search_results": search_results
        }
        
        # Add to appropriate split
        if i < calibration_samples:
            mcqa_data["calibration"].append(sample)
        else:
            mcqa_data["test"].append(sample)
    
    # Save output with 3 indents as requested
    print(f"\nSaving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mcqa_data, f, indent=3, ensure_ascii=False)
    
    print(f"\n=== Conversion Complete ===")
    print(f"✓ Created MCQA dataset with {total_samples} samples")
    print(f"✓ Calibration: {len(mcqa_data['calibration'])} samples")
    print(f"✓ Test: {len(mcqa_data['test'])} samples")
    print(f"✓ Output saved to: {output_file}")
    
    # Print some statistics about labels
    label_counts = {}
    for claim in all_claims:
        label = claim.get('label', 'NEI').upper()
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\n=== Label Distribution ===")
    for label, count in sorted(label_counts.items()):
        percentage = (count / total_samples) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")

def main():
    # Configuration
    healthver_directory = "./healthver"
    output_file = "healthver_mcqa.json"
    
    # Check if directory exists
    if not os.path.exists(healthver_directory):
        print(f"Directory not found: {healthver_directory}")
        print("Please make sure the HealthVer dataset is in the correct directory.")
        return
    
    # List files in directory
    print("Files in directory:")
    for file in os.listdir(healthver_directory):
        if file.endswith(('.json', '.jsonl')):
            filepath = os.path.join(healthver_directory, file)
            size = os.path.getsize(filepath)
            print(f" {file} ({size:,} bytes)")
    
    # Convert dataset
    convert_healthver_to_mcqa(healthver_directory, output_file)

if __name__ == "__main__":
    main()