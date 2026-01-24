"""RAG Query CLI - interact with the vector database and LLM."""

import json
import click

from rag_query.core.query_engine import Query


@click.group()
def cli():
    """RAG Query CLI - interact with the vector database and LLM."""
    # pass removed, docstring is enough


@cli.command()
@click.argument("question")
def query(question):
    """
    Print only the answer to the question.
    """
    query_engine = Query()
    answer = query_engine.get_answer(question)
    click.echo(answer)


@cli.command()
@click.argument("query_text")
def search(query_text):
    """
    Print the chunk texts and their metadata that are most similar to the query,
    with chunk numbers and spacing for readability.
    """
    query_engine = Query()
    all_data = query_engine.collection.get()
    all_docs = all_data.get("documents", [])
    # all_metas is unused, remove it or use it
    # all_metas = all_data.get("metadatas", [])

    click.echo(f"[DEBUG] Total chunks in DB: {len(all_docs)}")

    chunks = query_engine.get_relevant_chunks(query_text)
    documents = chunks.get("documents", [])
    metadatas = chunks.get("metadatas", [])

    found_any = False
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        if doc is not None:
            found_any = True
            click.echo(f"Found useful chunk {i+1}:\n{doc}\n")
            click.echo(f"Metadata: {json.dumps(meta, ensure_ascii=False, indent=2)}\n")

    if not found_any:
        click.echo("[DEBUG] No matching chunks found.")


if __name__ == "__main__":
    cli()
