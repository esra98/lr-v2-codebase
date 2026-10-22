"""Building the corpus from full-length recordings.

The stages, each resumable per session (``docs/audit_and_roadmap_2026-10-17.md``):

    ingest      videos/*.mp4            -> videos/sources.tsv
    transcribe  videos/<session>.mp4    -> work/<session>/transcript.json
    segment     transcript.json         -> work/<session>/segments.tsv
    cut         video + segments.tsv    -> dataset clips, work/<session>/clips.tsv
    build       clips.tsv of every session -> clip table, manifests

Every stage writes only under ``work/`` or the dataset folder, writes files
atomically, and records what it did in ``work/<session>/status.json``.
"""
