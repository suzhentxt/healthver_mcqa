import json
import os
import random
import pandas as pd
from typing import List, Dict, Any
from collections import Counter

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
        print(f"File not found: {filepath}")
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded JSON from {os.path.basename(filepath)}")
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

def load_csv_files(healthver_dir: str) -> Dict[str, str]:
    """Load CSV files and create a mapping from claim to question."""
    csv_files = ['healthver_dev.csv', 'healthver_train.csv', 'healthver_test.csv']
    claim_to_question = {}
    
    print("\n=== Loading CSV Files ===")
    for filename in csv_files:
        filepath = os.path.join(healthver_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
                print(f"Loaded {len(df)} rows from {filename}")
                
                print(f"Columns in {filename}: {list(df.columns)}")
                
                if 'claim' in df.columns and 'question' in df.columns:
                    for _, row in df.iterrows():
                        claim = str(row['claim']).strip()
                        question = str(row['question']).strip()
                        if claim and question and question != 'nan':
                            claim_to_question[claim] = question
                    print(f"Added {len(claim_to_question)} claim-question mappings from {filename}")
                else:
                    print(f"Warning: Missing 'claim' or 'question' columns in {filename}")
                    
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        else:
            print(f"CSV file not found: {filepath}")
    
    print(f"Total unique claim-question mappings: {len(claim_to_question)}")
    return claim_to_question

def determine_majority_label(claim: Dict[str, Any], corpus_dict: Dict[str, Dict]) -> str:
    """
    Determine the correct answer using majority label rule based on evidence field.
    
    Args:
        claim: Claim dictionary containing 'evidence' and 'doc_ids'
        corpus_dict: Dictionary mapping doc_id to document data (for fallback)
    
    Returns:
        correct_answer ("A", "B", "C") or None if claim should be skipped
    """
    doc_ids = claim.get('doc_ids', [])
    evidence = claim.get('evidence', {})
    
    if not doc_ids or not evidence:
        print(f"Warning: No doc_ids or evidence for claim id {claim.get('id', 'unknown')}")
        return None
    
    # Get labels from evidence field
    labels = []
    for doc_id in doc_ids:
        doc_id_str = str(doc_id)
        if doc_id_str in evidence:
            for evidence_item in evidence[doc_id_str]:
                doc_label = evidence_item.get('label', 'NEI')
                # Normalize label names
                if doc_label.upper() in ['SUPPORT', 'SUPPORTS']:
                    labels.append('SUPPORT')
                elif doc_label.upper() in ['CONTRADICT', 'REFUTES']:
                    labels.append('CONTRADICT')
                elif doc_label.upper() in ['NEI', 'NOT_ENOUGH_INFO']:
                    labels.append('NEI')
                else:
                    print(f"Warning: Unknown label '{doc_label}' for doc_id {doc_id}, treating as NEI")
                    labels.append('NEI')
        else:
            # Fallback to corpus_dict if evidence is missing for doc_id
            if doc_id_str in corpus_dict:
                doc_label = corpus_dict[doc_id_str].get('label', 'NEI')
                if doc_label.upper() in ['SUPPORT', 'SUPPORTS']:
                    labels.append('SUPPORT')
                elif doc_label.upper() in ['CONTRADICT', 'REFUTES']:
                    labels.append('CONTRADICT')
                elif doc_label.upper() in ['NEI', 'NOT_ENOUGH_INFO']:
                    labels.append('NEI')
                else:
                    print(f"Warning: Unknown label '{doc_label}' for doc_id {doc_id}, treating as NEI")
                    labels.append('NEI')
            else:
                print(f"Warning: doc_id {doc_id} not found in evidence or corpus, treating as NEI")
                labels.append('NEI')
    
    if not labels:
        print(f"Warning: No valid labels for claim id {claim.get('id', 'unknown')}")
        return None
    
    # Debug: Print labels for this claim
    print(f"Claim id {claim.get('id', 'unknown')}: doc_ids={doc_ids}, labels={labels}")
    
    # Apply majority label rule
    if len(labels) == 1:
        # Single doc: use its label
        label = labels[0]
    elif len(labels) == 2:
        # Two docs
        if labels[0] == labels[1]:
            # Same label: use that label
            label = labels[0]
        elif set(labels) == {'NEI', 'SUPPORT'} or set(labels) == {'NEI', 'CONTRADICT'}:
            # NEI vs SUPPORT/CONTRADICT: choose non-NEI
            label = 'SUPPORT' if 'SUPPORT' in labels else 'CONTRADICT'
        elif set(labels) == {'SUPPORT', 'CONTRADICT'}:
            # SUPPORT vs CONTRADICT: skip this claim
            print(f"Skipping claim id {claim.get('id', 'unknown')} due to SUPPORT vs CONTRADICT conflict")
            return None
        else:
            label = labels[0]  # Fallback
    else:
        # 3+ docs: majority vote
        label_counts = Counter(labels)
        most_common = label_counts.most_common()
        
        # Check for tie
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # Tie: skip this claim
            print(f"Skipping claim id {claim.get('id', 'unknown')} due to label tie: {label_counts}")
            return None
        
        label = most_common[0][0]
    
    # Map to correct_answer
    if label == 'SUPPORT':
        return "A"
    elif label == 'CONTRADICT':
        return "B"
    elif label == 'NEI':
        return "C"
    else:
        print(f"Warning: Invalid label '{label}' for claim id {claim.get('id', 'unknown')}")
        return None

def create_search_results(claim: Dict[str, Any], corpus_data: List[Dict]) -> List[Dict[str, Any]]:
    """Create search results for a claim using only relevant doc_ids."""
    search_results = []
    
    # Get doc_ids from the claim
    doc_ids = claim.get('doc_ids', [])
    
    # Create a mapping from doc_id to corpus document
    corpus_dict = {str(doc['doc_id']): doc for doc in corpus_data}  # Use doc_id as key
    
    # Get documents based on doc_ids
    for doc_id in doc_ids:  # Use all doc_ids
        doc_id_str = str(doc_id)  # Convert to string for consistency
        if doc_id_str in corpus_dict:
            doc = corpus_dict[doc_id_str]
            # page_name = title from corpus
            page_name = doc.get('title', f'Health Document {doc_id}')
            # page_snippet = abstract from corpus
            page_snippet = doc.get('abstract', ['Health-related content'])
            if isinstance(page_snippet, list):
                page_snippet = ' '.join(page_snippet)  # Join sentences into a single string
            else:
                page_snippet = str(page_snippet) if page_snippet else 'Health-related content'
        else:
            # Fallback if doc_id not found in corpus
            page_name = f'Health Document {doc_id}'
            page_snippet = 'Health-related content'
        
        search_result = {
            "page_name": page_name,
            "page_url": "",
            "page_snippet": page_snippet,
            "page_result": "",
            "page_last_modified": ""
        }
        search_results.append(search_result)
    
    return search_results

def convert_healthver_to_mcqa(healthver_dir: str, output_file: str):
    """Convert HealthVer dataset to MCQA format."""
    
    print(f"Looking for HealthVer files in: {healthver_dir}")
    
    # Load CSV files first to get claim-question mappings
    claim_to_question = load_csv_files(healthver_dir)
    
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
    
    # Load corpus (for search results and fallback labels)
    corpus_path = os.path.join(healthver_dir, 'corpus.jsonl')
    corpus_data = load_jsonl_file(corpus_path)
    
    if not corpus_data:
        print("Warning: corpus.jsonl not found or empty! Cannot create search results.")
        return
    
    # Debug: Print first corpus entry structure
    if corpus_data:
        print("\n=== Sample Corpus Structure ===")
        print(json.dumps(corpus_data[0], indent=2, ensure_ascii=False))
    
    # Create corpus dictionary for quick lookup
    corpus_dict = {str(doc['doc_id']): doc for doc in corpus_data}
    print(f"Created corpus dictionary with {len(corpus_dict)} documents")
    
    # Debug: Check specific doc_ids for claim id 0
    print("\n=== Checking corpus.jsonl for doc_ids [57, 72, 106, 328] ===")
    for doc_id in [57, 72, 106, 328]:
        doc_id_str = str(doc_id)
        if doc_id_str in corpus_dict:
            print(f"doc_id {doc_id}: label={corpus_dict[doc_id_str].get('label', 'No label')}")
        else:
            print(f"doc_id {doc_id}: Not found in corpus")
    
    # Process claims and filter valid ones
    valid_samples = []
    skipped_counts = {
        "no_doc_ids": 0,
        "majority_rule_skip": 0,
        "no_corpus_match": 0
    }
    
    # Track question source statistics
    question_sources = {"from_csv": 0, "generated": 0}
    
    # Track label distribution for final samples
    final_label_counts = {"A": 0, "B": 0, "C": 0}
    
    print(f"\n=== Processing Claims with Majority Label Rule ===")
    for i, claim in enumerate(all_claims):
        if i % 1000 == 0:
            print(f"Processing claim {i+1}/{len(all_claims)}")
        
        claim_text = claim.get('claim', 'No claim text')
        doc_ids = claim.get('doc_ids', [])
        
        # Skip claims without doc_ids
        if not doc_ids:
            skipped_counts["no_doc_ids"] += 1
            continue
        
        # Apply majority label rule using evidence
        correct_answer = determine_majority_label(claim, corpus_dict)
        
        # Skip claims that don't pass majority rule
        if correct_answer is None:
            skipped_counts["majority_rule_skip"] += 1
            continue
        
        # Get question from CSV mapping or generate default
        question_text = claim_to_question.get(claim_text.strip())
        if question_text:
            # Use question from CSV and add options
            question = f"{question_text}\nA. Supported\nB. Refuted\nC. Not Enough Information"
            question_sources["from_csv"] += 1
        else:
            # Generate default question
            question = f"Is the following health claim supported by evidence: {claim_text}\nA. Supported\nB. Refuted\nC. Not Enough Information"
            question_sources["generated"] += 1
        
        # Create search results
        search_results = create_search_results(claim, corpus_data)
        
        # Create sample with unique ID
        sample = {
            "id": f"{claim['source_split']}_{claim.get('id', i)}",  # Add split prefix
            "question": question,
            "correct_answer": correct_answer,
            "options": ["A", "B", "C"],
            "search_results": search_results
        }
        
        valid_samples.append(sample)
        final_label_counts[correct_answer] += 1
    
    print(f"\n=== Sample Processing Summary ===")
    print(f"Original claims: {len(all_claims)}")
    print(f"Valid samples after majority rule: {len(valid_samples)}")
    print(f"Skipped - no doc_ids: {skipped_counts['no_doc_ids']}")
    print(f"Skipped - majority rule conflicts: {skipped_counts['majority_rule_skip']}")
    print(f"Skipped - no corpus match: {skipped_counts['no_corpus_match']}")
    
    if len(valid_samples) == 0:
        print("No valid samples created! Please check your data.")
        return
    
    # Shuffle valid samples
    random.shuffle(valid_samples)
    
    # Create MCQA structure
    total_samples = len(valid_samples)
    calibration_samples = min(50, max(5, total_samples // 4))  # 25% or max 50, min 5
    
    mcqa_data = {
        "name": "HEALTHVER_MCQA",
        "description": "HealthVer dataset converted to MCQA format for evaluating health claim verification systems.",
        "version": "1.0",
        "total_samples": total_samples,
        "calibration_samples": calibration_samples,
        "test_samples": total_samples - calibration_samples,
        "calibration": [],
        "test": []
    }
    
    print(f"\n=== Creating MCQA Dataset ===")
    print(f"Total samples: {total_samples}")
    print(f"Calibration samples: {calibration_samples}")
    print(f"Test samples: {total_samples - calibration_samples}")
    
    # Split samples
    for i, sample in enumerate(valid_samples):
        if i < calibration_samples:
            mcqa_data["calibration"].append(sample)
        else:
            mcqa_data["test"].append(sample)
    
    # Save output with 3 indents
    print(f"\nSaving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mcqa_data, f, indent=3, ensure_ascii=False)
    
    print(f"\n=== Conversion Complete ===")
    print(f"✓ Created MCQA dataset with {total_samples} samples")
    print(f"✓ Calibration: {len(mcqa_data['calibration'])} samples")
    print(f"✓ Test: {len(mcqa_data['test'])} samples")
    print(f"✓ Output saved to: {output_file}")
    
    # Print question source statistics
    print(f"\n=== Question Source Statistics ===")
    print(f"Questions from CSV: {question_sources['from_csv']} ({question_sources['from_csv']/total_samples*100:.1f}%)")
    print(f"Generated questions: {question_sources['generated']} ({question_sources['generated']/total_samples*100:.1f}%)")
    
    # Print final label distribution
    print(f"\n=== Final Label Distribution (After Majority Rule) ===")
    for answer, count in sorted(final_label_counts.items()):
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        label_name = {"A": "SUPPORT", "B": "CONTRADICT", "C": "NEI"}[answer]
        print(f"{answer} ({label_name}): {count} ({percentage:.1f}%)")

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
        if file.endswith(('.json', '.jsonl', '.csv')):
            filepath = os.path.join(healthver_directory, file)
            size = os.path.getsize(filepath)
            print(f" {file} ({size:,} bytes)")
    
    # Convert dataset
    convert_healthver_to_mcqa(healthver_directory, output_file)

if __name__ == "__main__":
    main()