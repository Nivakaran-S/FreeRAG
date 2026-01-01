"""FreeRAG - A modular RAG system using Phi-3.5-mini.

CLI entrypoint for ingesting documents and querying the RAG system.
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from pathlib import Path

from src.config import Config
from src.rag.pipeline import RAGPipeline

app = typer.Typer(help="FreeRAG - Local RAG system with Phi-3.5-mini")
console = Console()


def get_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline."""
    return RAGPipeline(Config.default())


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to file or directory to ingest"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r", help="Recursively search directories")
):
    """Ingest documents into the vector store."""
    pipeline = get_pipeline()
    path_obj = Path(path)
    
    if not path_obj.exists():
        console.print(f"[red]Error: Path not found: {path}[/red]")
        raise typer.Exit(1)
    
    with console.status("[bold green]Ingesting documents..."):
        if path_obj.is_file():
            count = pipeline.ingest_file(path)
        else:
            count = pipeline.ingest_directory(path, recursive=recursive)
    
    console.print(Panel(f"[green]Successfully ingested {count} chunks![/green]"))


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of documents to retrieve")
):
    """Query the RAG system."""
    pipeline = get_pipeline()
    
    if pipeline.vector_store.get_count() == 0:
        console.print("[yellow]Warning: No documents in vector store. Run 'ingest' first.[/yellow]")
    
    with console.status("[bold green]Thinking..."):
        result = pipeline.query(question, top_k=top_k)
    
    # Display answer
    console.print(Panel(Markdown(result["answer"]), title="[bold blue]Answer[/bold blue]"))
    
    # Display sources
    if result["sources"]:
        console.print("\n[dim]Sources:[/dim]")
        for src in result["sources"]:
            console.print(f"  • {src['filename']}")


@app.command()
def chat():
    """Interactive chat mode."""
    pipeline = get_pipeline()
    
    console.print(Panel(
        "[bold]FreeRAG Chat Mode[/bold]\n"
        "Type your questions and press Enter.\n"
        "Type 'exit' or 'quit' to stop.",
        title="🤖 FreeRAG"
    ))
    
    doc_count = pipeline.vector_store.get_count()
    console.print(f"[dim]Loaded {doc_count} document chunks.[/dim]\n")
    
    while True:
        try:
            question = console.input("[bold blue]You:[/bold blue] ")
            
            if question.lower() in ["exit", "quit", "q"]:
                console.print("[dim]Goodbye![/dim]")
                break
            
            if not question.strip():
                continue
            
            with console.status("[bold green]Thinking..."):
                answer = pipeline.chat(question)
            
            console.print(f"[bold green]Assistant:[/bold green] {answer}\n")
            
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break


@app.command()
def stats():
    """Show vector store statistics."""
    pipeline = get_pipeline()
    stats = pipeline.get_stats()
    
    console.print(Panel(
        f"📊 [bold]Documents:[/bold] {stats['documents_count']} chunks\n"
        f"🗃️  [bold]Collection:[/bold] {stats['collection_name']}\n"
        f"🤖 [bold]LLM:[/bold] {stats['model']}\n"
        f"📐 [bold]Embeddings:[/bold] {stats['embedding_model']}",
        title="FreeRAG Statistics"
    ))


@app.command()
def clear():
    """Clear the vector store."""
    if typer.confirm("Are you sure you want to clear all documents?"):
        pipeline = get_pipeline()
        pipeline.vector_store.clear()
        console.print("[green]Vector store cleared.[/green]")


if __name__ == "__main__":
    app()
