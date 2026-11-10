"""Send configured model requests; use --help for required inputs."""

from send_vlm_request import main


if __name__ == "__main__":
    raise SystemExit(main(mode="concurrent", require_images=True))
