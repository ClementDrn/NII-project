from untitled.base import LOG
from untitled.vlm import VLM


### MAIN FUNCTION #############################################################
def main():
    # Init simulator

    # Init VLM
    vlm = VLM()

    # Run scenario once

    # Fetch results

    # Give data to VLM to generate instruction
    instruction = vlm.generate_instruction_from_images(data=None)
    LOG.info(f"Generated instruction: {instruction}")


    # Run scenario again with generated instruction

    # Fetch new results


if __name__ == "__main__":
    main()
