from __future__ import annotations

MENTAL_DISORDER_KEYWORDS = [
    "depression",
    "depressive",
    "dysthymia",
    "anxiety",
    "panic disorder",
    "phobia",
    "social anxiety",
    "ptsd",
    "post-traumatic",
    "stress disorder",
    "schizophrenia",
    "schizoaffective",
    "psychosis",
    "bipolar",
    "mania",
    "autism",
    "autistic",
    "adhd",
    "attention deficit",
    "hyperactivity",
    "obsessive-compulsive",
    "ocd",
    "eating disorder",
    "anorexia",
    "bulimia",
    "personality disorder",
    "borderline personality",
    "dementia",
    "alzheimer",
    "mental disorder",
    "psychiatric",
]

SEED_FAMILY_PATTERNS = {
    "depressive_disorders": [
        r"\bdepress",
        r"dysthymia",
        r"major depressive disorder",
    ],
    "anxiety_disorders": [
        r"\banxiety disorder\b",
        r"\bagoraphobia\b",
        r"\bpanic disorder\b",
        r"\bphobia\b",
        r"\bsocial phobia\b",
        r"\bspecific phobia\b",
    ],
    "trauma_stress_disorders": [
        r"\bptsd\b",
        r"post-traumatic stress disorder",
        r"acute stress disorder",
        r"occupation-related stress disorder",
    ],
    "psychotic_disorders": [
        r"schizophrenia",
        r"schizoaffective",
        r"\bpsychosis\b",
    ],
    "bipolar_disorders": [
        r"\bbipolar\b",
    ],
    "autism_adhd_disorders": [
        r"\bautism\b",
        r"attention deficit[- ]hyperactivity disorder",
        r"\badhd\b",
    ],
    "obsessive_compulsive_disorders": [
        r"obsessive-compulsive disorder",
        r"\bocd\b",
    ],
    "eating_disorders": [
        r"anorexia nervosa",
        r"bulimia nervosa",
        r"binge eating disorder",
    ],
    "personality_disorders": [
        r"personality disorder",
        r"multiple personality disorder",
    ],
    "neurocognitive_disorders": [
        r"alzheimer disease",
        r"frontotemporal dementia",
        r"lewy body dementia",
        r"cerebrovascular dementia",
        r"\bdementia\b",
    ],
}

SEED_EXCLUSION_PATTERNS = [
    r"susceptib",
    r"\bx-linked\b",
    r"intellectual disability",
    r"intellectual developmental disorder",
    r"developmental delay",
    r"\bcongenital\b",
    r"dysmorph",
    r"microceph",
    r"\bsyndrome\b",
    r"point mutation",
    r"microdeletion",
    r"\bepilepsy\b",
    r"amyotrophic lateral sclerosis",
    r"arthrogryposis",
    r"cachexia",
    r"arrhythmia",
    r"prion pathology",
    r"port-wine stain",
    r"hepatopathy",
    r"\(disease\)$",
    r"^anxiety$",
    r"^mental disorder$",
    r"drug/alcohol-induced mental disorder",
    r"developmental disorder of mental health",
]

MENTAL_NODE_TYPES = {"disease"}
PHENOTYPE_NODE_TYPES = {"effect/phenotype"}
DRUG_NODE_TYPES = {"drug"}
TARGET_NODE_TYPES = MENTAL_NODE_TYPES | PHENOTYPE_NODE_TYPES | DRUG_NODE_TYPES

RELATION_GROUP_RULES = {
    "disease_phenotype": {
        "type_pairs": {
            ("disease", "effect/phenotype"),
            ("effect/phenotype", "disease"),
        },
        "display_relations": {
            "phenotype present",
            "phenotype absent",
            "associated with",
        },
    },
    "disease_drug": {
        "type_pairs": {
            ("disease", "drug"),
            ("drug", "disease"),
        },
        "display_relations": {
            "indication",
            "contraindication",
            "off-label use",
        },
    },
    "disease_disease": {
        "type_pairs": {
            ("disease", "disease"),
        },
        "display_relations": {
            "associated with",
            "parent-child",
            "linked to",
        },
    },
}
