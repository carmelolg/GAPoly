from .cli import main

if __name__ == "__main__":  # required: worker processes re-import this module
    raise SystemExit(main())
