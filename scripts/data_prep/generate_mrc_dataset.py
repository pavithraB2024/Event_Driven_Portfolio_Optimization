import json
import random
import os

def generate_dataset(output_path, num_samples=100):
    """
    Generates a synthetic SQuAD-style dataset for FinBERT-MRC training based on Table 2 Schema.
    """
    
    # 1. Templates per Event Type matching Table 2
    
    # Structure: (News Template, [Questions...], [Role Map...])
    # Role Map keys map to placeholder names in template.
    
    templates = {
        "Mergers and Acquisitions": [
            (
                "{acquirer} has announced it will acquire {acquired} for ${amount} billion in an all-cash deal.",
                [
                    ("Which company is the acquirer?", "{acquirer}"),
                    ("Which company is being acquired?", "{acquired}"),
                    ("What is the deal size?", "${amount} billion")
                ]
            ),
            (
                "{acquired} shares surged after news that {acquirer} is planning a takeover bid.",
                [
                    ("Which company is the acquirer?", "{acquirer}"),
                    ("Which company is being acquired?", "{acquired}")
                ]
            )
        ],
        "Earnings Report": [
            (
                "{company} reported Q3 earnings of ${eps} per share, beating analyst expectations.",
                [
                    ("Which company reported earnings?", "{company}"),
                    ("What is the reported EPS?", "${eps}"),
                    ("Did it beat expectations?", "beating analyst expectations") # Text span
                ]
            ),
            (
                "{company} stock fell as revenue missed targets, despite an EPS of ${eps}.",
                [
                    ("Which company reported earnings?", "{company}"),
                    ("What is the reported EPS?", "${eps}"),
                    ("Did it beat expectations?", "missed targets")
                ]
            )
        ],
        "Legal and Regulation": [
            (
                "Regulators have fined {company} ${amount} million for antitrust violations.",
                [
                    ("Which company was fined?", "{company}"),
                    ("What is the penalty amount?", "${amount} million"),
                    ("Why was the company fined?", "antitrust violations")
                ]
            ),
             (
                "{company} is facing a lawsuit from {plaintiff} regarding patent infringement.",
                [
                    ("Which company is sued or fined?", "{company}"),
                    ("Who is the plaintiff?", "{plaintiff}")
                ]
            )
        ],
        "Supply Chain": [
             (
                "{company} warned of production delays due to component shortages from {supplier}.",
                [
                    ("Which company is facing delays?", "{company}"),
                    ("Which supplier is causing delays?", "{supplier}")
                ]
            )
        ]
    }
    
    # 2. Entity Pools
    companies = ["Apple", "Microsoft", "Google", "Amazon", "Tesla", "Boeing", "Pfizer", "Intel", "Cisco", "Oracle"]
    amounts = ["1.2", "4.5", "10", "25", "0.5"]
    eps_vals = ["1.20", "3.45", "0.89", "5.12"]
    suppliers = ["TSMC", "Foxconn", "Samsung", "LG", "Bosch"]
    
    data = []
    
    for _ in range(num_samples):
        # Pick random event type
        etype = random.choice(list(templates.keys()))
        template_group = templates[etype]
        template_str, questions = random.choice(template_group)
        
        # Fill placeholders
        fillers = {
            "acquirer": random.choice(companies),
            "acquired": random.choice([c for c in companies if c != "Apple"]), # Avoid same
            "company": random.choice(companies),
            "plaintiff": random.choice(["Department of Justice", "FTC", "EU Commission", "Epic Games"]),
            "supplier": random.choice(suppliers),
            "amount": random.choice(amounts),
            "eps": random.choice(eps_vals)
        }
        
        # Ensure unique A/B
        if "{acquirer}" in template_str and fillers["acquirer"] == fillers["acquired"]:
             fillers["acquired"] = "Target Corp"
             
        # Create Context
        context = template_str.format(**fillers)
        
        qas = []
        for q_text, ans_placeholder in questions:
            # Resolve answer text
            if ans_placeholder.startswith("{") and ans_placeholder.endswith("}"):
                # Extract key "acquirer" from "{acquirer}"
                key = ans_placeholder[1:-1]
                if key in fillers:
                    ans_text = fillers[key]
                else:
                    # Might be a complex template like "${amount} billion" logic? 
                    # Actually valid case is ans_placeholder="{company}" -> key="company"
                    ans_text = ans_placeholder.format(**fillers)
            else:
                 # It might be a template string itself like "${amount} billion"
                 ans_text = ans_placeholder.format(**fillers)
            
            # Find start position
            try:
                start_idx = context.index(ans_text)
                
                qas.append({
                    "id": str(random.getrandbits(64)),
                    "question": q_text,
                    "answers": [
                        {
                            "text": ans_text,
                            "answer_start": start_idx
                        }
                    ],
                    "is_impossible": False
                })
            except ValueError:
                # Should not happen with well-formed templates
                continue
                
        if qas:
            data.append({
                "title": etype,
                "paragraphs": [
                    {
                        "context": context,
                        "qas": qas
                    }
                ]
            })
            
    # Wrap in SQuAD format
    squad_format = {"version": "2.0", "data": data}
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(squad_format, f, indent=2)
        
    print(f"Generated {len(data)} synthetic samples at {output_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "finbert_mrc_train_synthetic.json")
    
    # Clean up old dir structure assumption
    # script is in scripts/, so BASE is root.
    # data/ is at root.
    
    generate_dataset(OUTPUT_FILE, num_samples=200)
