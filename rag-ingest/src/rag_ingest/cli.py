"""CLI for RAG Ingester."""

import sys
from pathlib import Path

import click

from rag_ingest.core.config import load_config
from rag_ingest.core.ingestion_pipeline import IngestionPipeline
from rag_ingest.core.source_manager import SourceManager
from rag_ingest.core.vector_db import VectorDB


@click.group()
def cli():
    """RAG Ingester CLI."""


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--method",
    type=click.Choice(["1", "2", "3", "4", "5"], case_sensitive=False),
    default="1",
    help="Metoda de chunking: 1 = clasic, 2 = AI-enhanced, " \
    "3 = semantic, 4 = markdown, 5 = recursive",
)
def ingest(file_path, method):
    """Ingest a single document, returns document ID."""
    try:
        pipeline = IngestionPipeline()
        doc_id = pipeline.ingest_document(file_path, method=method)
        click.echo(f"Document ingested with ID: {doc_id}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        click.echo(f"Error ingesting document: {e}", err=True)
        sys.exit(1)


@cli.command("ingest-batch")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--method",
    type=click.Choice(["1", "2"], case_sensitive=False),
    default="1",
    help="Metoda de chunking: 1 = clasic, 2 = AI-enhanced",
)
def ingest_batch(directory, method):
    """Ingest all documents in a directory."""
    try:
        pipeline = IngestionPipeline()
        file_paths = [str(p) for p in Path(directory).glob("*") if p.is_file()]
        doc_ids = pipeline.batch_ingest(file_paths, method)
        click.echo(f"Batch ingested document IDs: {doc_ids}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        click.echo(f"Error in batch ingestion: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
def delete(file_path):
    """Delete document by computing checksum from file path."""
    try:
        pipeline = IngestionPipeline()
        result = pipeline.delete_document_by_path(file_path)
        if result:
            click.echo("Document deleted successfully.")
        else:
            click.echo("Document not found or could not be deleted.", err=True)
            sys.exit(1)
    except Exception as e:  # pylint: disable=broad-exception-caught
        click.echo(f"Error deleting document: {e}", err=True)
        sys.exit(1)


@cli.command("delete-id")
@click.argument("document_id")
def delete_id(document_id):
    """Delete document by checksum ID directly."""
    try:
        pipeline = IngestionPipeline()
        result = pipeline.delete_document(document_id)
        if result:
            click.echo("Document deleted successfully.")
        else:
            click.echo("Document not found or could not be deleted.", err=True)
            sys.exit(1)
    except Exception as e:  # pylint: disable=broad-exception-caught
        click.echo(f"Error deleting document by ID: {e}", err=True)
        sys.exit(1)


@cli.command(name="list")
def list_documents():  # renamed to avoid built-in shadowing
    """List all ingested documents with their checksums and metadata."""
    try:
        config = load_config()
        smanager = SourceManager(config.storage_dir)
        docs = smanager.list_documents()
        for doc in docs:
            click.echo(doc)
    except Exception as e:  # pylint: disable=broad-exception-caught
        click.echo(f"Error listing documents: {e}", err=True)
        sys.exit(1)


@cli.command("count")
def get_number():
    """Get the number of documents in the vector database."""
    try:
        number = VectorDB.get_document_count()
        click.echo(number)
    except Exception as e:  # pylint: disable=broad-exception-caught
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
