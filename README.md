
# The International Student Survival Gui

A Retrieval-Augmented Generation (RAG) system that answers practical questions about studying in the USA as an international student, based on real Reddit posts from r/InternationalStudents and r/f1visa.

## Domain

The domain is unofficial survival knowledge for international students in the USA — the real, practical information students share with each other on Reddit, but that never appears on any university website.

This knowledge is valuable because it covers what no official source will tell you: how brutal the OPT job market really is, how to get affordable health insurance, what work F-1 students can legally do, and how to actually make friends when everyone already knows each other.

## Document Sources

| # | Source | Type | URL or file path |
| --- | ------ | ---- | ---------------- |
| 1  | r/InternationalStudents | Reddit thread | documents/doc1.txt — first semester mistakes |
| 2  | r/InternationalStudents | Reddit thread | documents/doc2.txt — tips before coming to USA |
| 3  | r/InternationalStudents | Reddit thread | documents/doc3.txt — US safety fears |
| 4  | r/InternationalStudents | Reddit thread | documents/doc4.txt — teacher warning post |
| 5  | r/InternationalStudents | Reddit thread | documents/doc5.txt — how students afford travel |
| 6  | r/InternationalStudents | Reddit thread | documents/doc6.txt — F1 visa work restrictions |
| 7  | r/InternationalStudents | Reddit thread | documents/doc7.txt — cost of US education |
| 8  | r/f1visa | Reddit thread | documents/doc8.txt — health insurance for F1/OPT |
| 9  | r/InternationalStudents | Reddit thread | documents/doc9.txt — F1/OPT/H1B job market |
| 10 | r/Advice + r/InternationalStudents | Reddit thread | documents/doc10.txt — cultural shock and friends |

## Chunking Strategy

**Chunk size:** 400 characters

**Overlap:** 75 characters

**Why these choices fit your documents:**

Most of the documents are short Reddit comments where one person gives one piece of advice in 1-3 sentences. Key facts are usually concentrated in a single sentence. A chunk size of 400 characters fits most Reddit comments as one complete thought.

An overlap of 75 characters means each chunk shares context with the next one.

**Final chunk count:** 97 chunks across 10 documents.

## Sample Chunks

**Chunk from doc1.txt:** Did not explore scholarships opening next semester. Honors scholarships can be up to 5000 dollars. Did not join enough clubs. Should have gotten an on-campus job early to get SSN.

**Chunk from doc6.txt:** Legal ways to earn money on F1 visa: On-campus employment up to 20 hours per week. CPT for work related to your major. OPT after graduation.

**Chunk from doc8.txt:** Affordable options for F1 students: ISO insurance around 70 dollars per month for basic coverage. PSI insurance similar to ISO. Compass insurance another popular option.

**Chunk from doc9.txt:** Apply for OPT up to 90 days before graduation not after. Start applying for jobs 6 months before graduation. Target smaller companies that are more flexible about visa situations.

**Chunk from doc10.txt:** Start small. Look around and see if you notice another student who seems away from the group. Smile at them, and if they smile back, make small talk.

## Embedding Model

**Model used:** all-MiniLM-L6-v2 (via sentence-transformers, runs locally)

**Production tradeoff reflection:**

I used all-MiniLM-L6-v2 because it runs locally with no API key and no rate limits — perfect for a student project. If deployed for real users with no cost constraint, I would consider OpenAI text-embedding-3-large for better semantic accuracy, Cohere multilingual embeddings for international students querying in native languages, or larger sentence-transformers like all-mpnet-base-v2 for higher quality at 3x slower speed.

## Grounded Generation

**System prompt grounding instruction:**

The LLM is given this instruction: "Answer the user's question using ONLY the context below. If the context does not contain enough information, say: The documents do not provide enough information to answer this fully."

**How source attribution is surfaced:**

After generation, the system programmatically appends a Retrieved from section showing the source filename, chunk index, and distance score. Citations are guaranteed by the pipeline.

## Evaluation Report

| # | Question | Expected answer | System response | Retrieval | Accuracy |
| --- | -------- | --------------- | --------------- | --------- | -------- |
| 1 | What are the biggest mistakes international students make in their first semester? | Wrong accommodation, overspending, not networking, poor time management | Listed wrong accommodation, overspending, not networking, poor time management, ignoring scholarships | Relevant | Accurate |
| 2 | How can I build credit in the US without a SSN? | Get ITIN, secured cards, cross-border banks | The documents do not provide enough information to answer this fully | Off-target | Inaccurate (correct refusal) |
| 3 | What work can I legally do on an F-1 student visa? | On-campus 20 hrs, CPT, OPT 12 or 36 months for STEM | Said 3 years under OPT — missed CPT and on-campus rules | Partially relevant | Partially accurate |
| 4 | How hard is it to find a job after graduation on OPT? | Very hard, companies filter out visa applicants, apply early | Explained visa checkbox problem and gave correct strategies | Relevant | Accurate |
| 5 | How do I deal with cultural shock and make friends in the US? | Start small, join clubs, give it time | Described starting small with smiles and small talk | Relevant | Accurate |

**Summary:** 3 accurate, 1 partially accurate, 1 honest refusal.

## Failure Case Analysis

**Question that failed:** How can I build credit in the US without a SSN?

**What the system returned:** The documents do not provide enough information to answer this fully. Retrieved chunks had distance scores 1.055, 1.315, 1.323, 1.434 — all above 1.0.

**Root cause:**

This is a document coverage failure, not a retrieval or generation failure. None of my 10 source documents specifically discuss building credit without an SSN. When I built the document collection in Milestone 1, I focused on broader topics like mistakes, visa rules, jobs, insurance, and social life — but did not include a dedicated source on credit-building. The high distance scores correctly indicated weak matches, and the grounded prompt told the system to refuse rather than hallucinate.

**What I would change to fix it:**

Add at least one document specifically about banking and credit for international students. A second advanced fix would be hybrid search combining semantic search with BM25 keyword matching, which would catch exact terms like SSN and credit even when semantic similarity scored them weakly.

## Spec Reflection

**One way the spec helped:**

Writing the chunking strategy in planning.md before writing code forced me to think about what my documents actually looked like. I noticed Reddit comments concentrate facts in single sentences. This led me to pick 400 characters and 75 overlap, which worked well — 97 chunks with most queries returning relevant results.

**One way the implementation diverged:**

My spec said I would use LangChain RecursiveCharacterTextSplitter, but I switched to a simpler manual chunking function. LangChain was not in the starter requirements.txt and adding it would have required extra dependencies. Manual chunking with text slicing worked fine and kept dependencies minimal.

## AI Usage

**Instance 1**

- *What I gave the AI:* My planning.md chunking strategy and asked it to write ingest.py.
- *What it produced:* A script that loads .txt files, cleans them with regex, chunks them with 400/75 overlap, embeds with all-MiniLM-L6-v2, and stores in ChromaDB.
- *What I changed:* The AI initially used LangChain text splitter requiring extra dependencies. I asked it to rewrite using only the packages in my requirements.txt.

**Instance 2**

- *What I gave the AI:* My grounding requirement and asked for rag.py and app.py.
- *What it produced:* rag.py with retrieval and generation logic, and app.py with a Gradio interface and 5 example questions.
- *What I changed:* The AI initially used llama-3.1-8b-instant, but my planning.md specified llama-3.3-70b-versatile. I changed it to match the spec. I also added distance scores to the source output so retrieval quality shows in the UI.