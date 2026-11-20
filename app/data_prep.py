"""Data preparation for Legal Document LoRA Tagger."""

import random
import logging
import math

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object  # fallback base

try:
    from datasets import load_dataset as hf_load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from transformers import PreTrainedTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Clause labels
# ---------------------------------------------------------------------------

CLAUSE_LABELS = [
    "Governing_Law",
    "Termination",
    "Confidentiality",
    "Indemnification",
    "Limitation_of_Liability",
    "Dispute_Resolution",
    "Intellectual_Property",
    "Payment_Terms",
    "Force_Majeure",
    "Assignment",
    "Warranties",
    "Non_Compete",
    "Data_Protection",
    "Notice",
    "Amendment",
    "Waiver",
    "Severability",
    "Entire_Agreement",
    "Counterparts",
    "Survival",
]

# ---------------------------------------------------------------------------
# 2. Sample data — at least 3 examples per label
# ---------------------------------------------------------------------------

SAMPLE_DATA = {
    "Governing_Law": [
        "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles.",
        "The parties agree that this contract shall be governed by the laws of England and Wales.",
        "This Agreement and any dispute arising hereunder shall be governed by the laws of the Republic of India.",
        "All matters relating to this Agreement shall be governed by the federal laws of the United States of America.",
        "This contract shall be interpreted under and subject to the laws of the Province of Ontario, Canada.",
    ],
    "Termination": [
        "Either party may terminate this Agreement upon thirty (30) days' prior written notice to the other party.",
        "This Agreement may be terminated immediately by either party in the event of a material breach that remains uncured for fifteen (15) days after written notice.",
        "The Company reserves the right to terminate this Agreement for convenience upon sixty (60) days' written notice.",
        "Upon termination of this Agreement for any reason, all licenses granted hereunder shall immediately cease and terminate.",
        "Either party may terminate this Agreement without cause upon ninety (90) days' written notice.",
    ],
    "Confidentiality": [
        "Each party agrees to keep confidential all proprietary information disclosed by the other party and shall not disclose such information to any third party without prior written consent.",
        "The Receiving Party shall use the Confidential Information solely for the purposes contemplated by this Agreement and shall protect it with the same degree of care it uses for its own confidential information.",
        "All confidential information exchanged under this Agreement shall remain confidential for a period of five (5) years following the termination of this Agreement.",
        "The obligations of confidentiality shall not apply to information that is or becomes publicly known through no breach of this Agreement.",
        "The Recipient shall promptly notify the Discloser upon becoming aware of any unauthorized disclosure or use of the Confidential Information.",
    ],
    "Indemnification": [
        "The Company shall indemnify, defend, and hold harmless the Client from and against any claims, damages, losses, or expenses arising out of the Company's breach of this Agreement.",
        "Each party shall indemnify the other party against all third-party claims arising from that party's negligence or willful misconduct.",
        "The Vendor shall indemnify the Purchaser from any intellectual property infringement claims arising from the use of the delivered products.",
        "Client shall indemnify and hold harmless the Service Provider from any claims arising from Client's use of the services in violation of applicable law.",
        "The indemnifying party shall promptly assume control of the defense of any indemnified claim and shall not settle any such claim without the indemnified party's prior written consent.",
    ],
    "Limitation_of_Liability": [
        "In no event shall either party be liable for any indirect, incidental, special, exemplary, or consequential damages arising out of or related to this Agreement.",
        "The aggregate liability of the Company under this Agreement shall not exceed the total fees paid by the Client in the twelve (12) months preceding the claim.",
        "Neither party shall be liable for lost profits, lost revenue, or loss of data under any theory of liability.",
        "The limitations of liability set forth in this section shall apply notwithstanding any failure of essential purpose of any limited remedy.",
        "In no event shall either party's liability arising under this Agreement exceed one hundred thousand dollars ($100,000).",
    ],
    "Dispute_Resolution": [
        "Any dispute arising out of or in connection with this Agreement shall be settled by binding arbitration under the rules of the American Arbitration Association.",
        "The parties agree to first attempt to resolve any dispute through good-faith negotiation before resorting to formal arbitration or litigation.",
        "Any controversy or claim arising out of this Agreement shall be submitted to arbitration in New York, New York, before a single arbitrator.",
        "In the event of a dispute, the parties shall submit the matter to mediation prior to commencing any legal proceedings.",
        "All disputes shall be resolved exclusively in the courts of competent jurisdiction in the city of London, England.",
    ],
    "Intellectual_Property": [
        "All intellectual property rights in and to any work product created by the Contractor under this Agreement shall be owned exclusively by the Client.",
        "The Company retains all right, title, and interest in its pre-existing intellectual property, including patents, trademarks, and copyrights.",
        "Any inventions, discoveries, or innovations developed during the course of this engagement shall be assigned to the Client upon creation.",
        "Each party grants the other a limited, non-exclusive license to use its intellectual property solely as necessary to perform this Agreement.",
        "The Contractor shall not use the Client's intellectual property for any purpose beyond the scope of this Agreement.",
    ],
    "Payment_Terms": [
        "Payment shall be due within thirty (30) days of receipt of a valid invoice from the Service Provider.",
        "The Client agrees to pay the agreed fees within fifteen (15) business days following the delivery of the services.",
        "All invoices unpaid after the due date shall accrue interest at the rate of one and a half percent (1.5%) per month.",
        "The Company shall invoice the Client on a monthly basis, and payment shall be due within forty-five (45) days of invoice date.",
        "Any disputed amounts must be communicated in writing within ten (10) business days of receipt of the invoice.",
    ],
    "Force_Majeure": [
        "Neither party shall be liable for any failure or delay in performance due to causes beyond its reasonable control, including natural disasters, acts of war, or government actions.",
        "In the event of a force majeure event, the affected party shall promptly notify the other party and take all reasonable steps to mitigate the impact.",
        "If a force majeure event continues for more than sixty (60) days, either party may terminate this Agreement without liability.",
        "Force majeure events include, but are not limited to, earthquakes, floods, pandemics, strikes, and government-imposed restrictions.",
        "The existence of a force majeure event shall suspend the obligations of the affected party for the duration of the event.",
    ],
    "Assignment": [
        "Neither party may assign its rights or obligations under this Agreement without the prior written consent of the other party.",
        "The Company may assign this Agreement to any of its affiliates or in connection with a merger, acquisition, or sale of all or substantially all of its assets.",
        "Any purported assignment in violation of this section shall be null and void.",
        "The Client may not transfer or assign any license granted under this Agreement to any third party without prior written approval.",
        "This Agreement shall be binding upon and inure to the benefit of the parties and their respective successors and permitted assigns.",
    ],
    "Warranties": [
        "The Company warrants that the services will be performed in a professional and workmanlike manner consistent with industry standards.",
        "Each party represents and warrants that it has full authority to enter into and perform its obligations under this Agreement.",
        "The Vendor warrants that the deliverables shall conform to the specifications set forth in Schedule A for a period of ninety (90) days.",
        "EXCEPT AS EXPRESSLY SET FORTH HEREIN, THE SERVICES ARE PROVIDED 'AS IS' WITHOUT WARRANTY OF ANY KIND.",
        "The Company does not warrant that the services will be error-free or uninterrupted.",
    ],
    "Non_Compete": [
        "During the term of this Agreement and for a period of two (2) years thereafter, the Employee shall not engage in any business that competes with the Company.",
        "The Contractor agrees not to solicit the Company's clients or employees during the term and for twelve (12) months following termination.",
        "The non-compete restrictions set forth herein shall apply within a radius of fifty (50) miles of any location where the Company conducts business.",
        "Employee shall not, directly or indirectly, become an officer, director, or partner of any competing business during the restricted period.",
        "In the event of a breach of this non-compete clause, the Company shall be entitled to seek injunctive relief without the need to post bond.",
    ],
    "Data_Protection": [
        "Each party shall comply with all applicable data protection laws, including the General Data Protection Regulation (GDPR), in connection with the processing of personal data.",
        "The Data Processor shall implement appropriate technical and organizational measures to protect personal data against unauthorized access or disclosure.",
        "Personal data shared under this Agreement shall be used solely for the purposes set out herein and shall not be retained beyond the period necessary.",
        "In the event of a personal data breach, the Data Processor shall notify the Data Controller within seventy-two (72) hours of becoming aware of the breach.",
        "The parties agree to execute a separate Data Processing Agreement where required by applicable law.",
    ],
    "Notice": [
        "All notices required or permitted under this Agreement shall be in writing and delivered by certified mail, overnight courier, or email with read receipt.",
        "Notice shall be deemed given upon receipt if delivered personally, three (3) business days after mailing, or upon confirmed email delivery.",
        "Any notice under this Agreement shall be addressed to the parties at the addresses set forth on the signature page.",
        "Either party may change its notice address by providing written notice to the other party at least ten (10) days in advance.",
        "Notices sent by email shall be deemed received only upon acknowledgment of receipt by the receiving party.",
    ],
    "Amendment": [
        "This Agreement may not be amended or modified except by a written instrument signed by authorized representatives of both parties.",
        "No amendment to this Agreement shall be effective unless it is in writing and duly executed by both parties.",
        "Any oral modification or amendment to this Agreement shall be of no force or effect.",
        "The parties may amend the Schedules to this Agreement by mutual written consent without amending the main body of the Agreement.",
        "Any waiver or modification of any term of this Agreement must be made in writing and signed by both parties to be effective.",
    ],
    "Waiver": [
        "The failure of either party to enforce any provision of this Agreement shall not constitute a waiver of that party's right to enforce such provision in the future.",
        "No waiver of any breach or default shall be deemed a waiver of any subsequent breach or default of the same or any other provision.",
        "A waiver is only effective if it is in writing and signed by the waiving party.",
        "Any delay or omission in exercising any right or remedy shall not impair or be construed as a waiver of such right or remedy.",
        "The rights and remedies of the parties under this Agreement are cumulative and not exclusive of any other rights or remedies.",
    ],
    "Severability": [
        "If any provision of this Agreement is found to be invalid, illegal, or unenforceable, the remaining provisions shall continue in full force and effect.",
        "In the event that any court of competent jurisdiction holds any provision of this Agreement to be unenforceable, such provision shall be modified to the minimum extent necessary to make it enforceable.",
        "The invalidity or unenforceability of any provision shall not affect the validity or enforceability of any other provision of this Agreement.",
        "If any provision is determined to be void or unenforceable, the parties shall negotiate in good faith a replacement provision that most closely reflects the original intent.",
        "The severability of provisions shall not be construed to limit the applicability of the remainder of this Agreement.",
    ],
    "Entire_Agreement": [
        "This Agreement constitutes the entire agreement between the parties with respect to its subject matter and supersedes all prior agreements, understandings, and negotiations.",
        "This Agreement, together with its Schedules and Exhibits, represents the complete and exclusive statement of the agreement between the parties.",
        "No prior drafts, representations, or other communications shall be used to interpret or supplement this Agreement.",
        "This Agreement supersedes all prior and contemporaneous discussions, negotiations, and agreements between the parties relating to the subject matter hereof.",
        "Any prior agreements, oral or written, regarding the subject matter of this Agreement are hereby terminated and replaced by this Agreement.",
    ],
    "Counterparts": [
        "This Agreement may be executed in one or more counterparts, each of which shall be deemed an original, and all of which together shall constitute one and the same instrument.",
        "Electronic signatures shall be deemed valid and binding to the same extent as original signatures.",
        "Delivery of an executed counterpart by electronic means shall be equally effective as delivery of a manually executed counterpart.",
        "Facsimile or PDF copies of signed counterparts shall have the same legal effect as original signed documents.",
        "The parties agree that electronic execution of this Agreement via DocuSign or similar services is valid and enforceable.",
    ],
    "Survival": [
        "The provisions of this Agreement relating to confidentiality, indemnification, limitation of liability, and governing law shall survive the termination or expiration of this Agreement.",
        "All rights and obligations that by their nature are intended to survive termination shall continue in effect after expiration of this Agreement.",
        "The following sections shall survive any termination of this Agreement: Sections 5, 6, 8, and 11.",
        "Survival of obligations under this Agreement shall not create any obligation to continue performance beyond termination.",
        "Any clause that by its nature should survive the expiration or termination of this Agreement shall so survive for a period of three (3) years.",
    ],
}

# ---------------------------------------------------------------------------
# 3. get_sample_dataset
# ---------------------------------------------------------------------------

def get_sample_dataset(n_per_class: int = 5, seed: int = 42):
    """
    Build a dataset from SAMPLE_DATA and split 70/15/15 stratified by label.
    Returns (train_list, val_list, test_list).
    Each item is {"text": str, "label": str}.
    """
    rng = random.Random(seed)

    all_items = []
    for label, examples in SAMPLE_DATA.items():
        # Cycle examples if n_per_class > available samples
        cycled = []
        while len(cycled) < n_per_class:
            cycled.extend(examples)
        cycled = cycled[:n_per_class]
        rng.shuffle(cycled)
        for text in cycled:
            all_items.append({"text": text, "label": label})

    # Stratified split per label
    train_list, val_list, test_list = [], [], []
    for label in CLAUSE_LABELS:
        items = [it for it in all_items if it["label"] == label]
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, math.floor(0.70 * n))
        n_val = max(1, math.floor(0.15 * n))
        # Ensure no overlap: train | val | test
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        train_list.extend(items[:n_train])
        val_list.extend(items[n_train:n_train + n_val])
        test_list.extend(items[n_train + n_val:])

    rng.shuffle(train_list)
    rng.shuffle(val_list)
    rng.shuffle(test_list)

    return train_list, val_list, test_list


# ---------------------------------------------------------------------------
# 4. try_load_ledgar
# ---------------------------------------------------------------------------

def try_load_ledgar(cache_dir: str = "./data/ledgar"):
    """
    Try to load LEDGAR from HuggingFace lex_glue.
    Returns (train_list, val_list, test_list) or None on failure.
    """
    if not HAS_DATASETS:
        logger.warning("'datasets' library not installed — LEDGAR unavailable.")
        return None

    try:
        logger.info("Attempting to load LEDGAR (lex_glue/ledgar) from HuggingFace...")
        raw = hf_load_dataset("lex_glue", "ledgar", cache_dir=cache_dir)
    except Exception as e:
        logger.warning(f"LEDGAR not available: {e}")
        return None

    # Map LEDGAR label names to our CLAUSE_LABELS where possible
    ledgar_label_names = raw["train"].features["label"].names if hasattr(raw["train"].features.get("label", None), "names") else []
    label_map = {}
    for idx, name in enumerate(ledgar_label_names):
        normalized = name.replace(" ", "_").replace("-", "_")
        for our_label in CLAUSE_LABELS:
            if our_label.lower() == normalized.lower():
                label_map[idx] = our_label
                break

    def convert_split(split_data):
        result = []
        for item in split_data:
            raw_label = item["label"]
            mapped = label_map.get(raw_label)
            if mapped is None:
                continue
            result.append({"text": item["text"], "label": mapped})
        return result

    try:
        train_list = convert_split(raw["train"])
        val_list = convert_split(raw["validation"])
        test_list = convert_split(raw["test"])
        if not train_list:
            logger.warning("LEDGAR loaded but no labels mapped to our CLAUSE_LABELS.")
            return None
        logger.info(f"LEDGAR loaded: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")
        return train_list, val_list, test_list
    except Exception as e:
        logger.warning(f"Error processing LEDGAR splits: {e}")
        return None


# ---------------------------------------------------------------------------
# 5. load_dataset_for_training
# ---------------------------------------------------------------------------

def load_dataset_for_training(
    use_ledgar: bool = False,
    n_per_class: int = 5,
    seed: int = 42,
    cache_dir: str = "./data",
):
    """
    Load data for training.
    Returns (train_list, val_list, test_list, source).
    source is "ledgar" or "sample".
    """
    if use_ledgar:
        result = try_load_ledgar(cache_dir=cache_dir)
        if result is not None:
            train, val, test = result
            return train, val, test, "ledgar"
        else:
            logger.warning("LEDGAR unavailable — falling back to built-in sample data.")

    train, val, test = get_sample_dataset(n_per_class=n_per_class, seed=seed)
    return train, val, test, "sample"


# ---------------------------------------------------------------------------
# 6. LegalClauseDataset
# ---------------------------------------------------------------------------

class LegalClauseDataset(Dataset if HAS_TORCH else object):
    """PyTorch Dataset for legal clause classification."""

    def __init__(self, data: list, tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"{item['text']} <{item['label']}>"
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
