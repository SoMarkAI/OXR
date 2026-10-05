import typer
from oxr.cli.cmd_serve import serve

app = typer.Typer(help="OXR: Optical Everything Recognition")


@app.callback()
def main():
    # Keep the explicit `serve` subcommand when it is the only command.
    pass


app.command(name="serve", help="Start the OXR server with its Online Demo at /demo")(serve)

if __name__ == "__main__":
    app()
