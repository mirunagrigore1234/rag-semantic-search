"""Module for extracting text from images using OpenAI vision model."""

import base64
from typing import List

from openai import OpenAI

from .base_extractor import BaseExtractor


class ImageBatch:
    """Represents a batch of images with an optional previous image for context."""
    # pylint: disable=too-few-public-methods

    def __init__(self, previous_image_path: str, image_paths: List[str]):
        self.previous_image_path = previous_image_path
        self.image_paths = image_paths

    def __str__(self):
        return f"previous_image_path={self.previous_image_path}, image_paths={self.image_paths}"


class ImageBatches:
    """Helper class to split image paths into overlapping batches."""
    # pylint: disable=too-few-public-methods

    def __init__(self, image_paths: List[str], batch_size: int):
        self.image_paths = image_paths
        self.batch_size = batch_size

    def get_batches(self) -> List[ImageBatch]:
        """
        Generate batches of images with overlap of last image from previous batch.

        Returns:
            List[ImageBatch]: List of image batches with context image.
        """
        batches = []
        previous_image_path = None
        for i in range(0, len(self.image_paths), self.batch_size):
            batch = self.image_paths[i : i + self.batch_size]
            batches.append(ImageBatch(previous_image_path, batch))
            previous_image_path = batch[-1] if batch else None
        return batches


class VisionExtractor(BaseExtractor):
    """
    Extract text from images using OpenAI's vision model.
    """

    @classmethod
    def extract(cls, file_path: str) -> str:  # pylint: disable=arguments-differ
        """
        Extract text from a single image file.

        Args:
            file_path (str): Path to the image file.

        Returns:
            str: Extracted text content.
        """
        with open(file_path, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode("utf-8")

        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all text from image in the original language, "
                                "and don't change or add anything."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ],
        )

        extracted_text = response.choices[0].message.content
        return extracted_text

    @staticmethod
    def extract_images_to_markdown(
        image_paths: List[str], batch_size: int = 4
    ) -> str:
        # pylint: disable=too-many-locals, broad-exception-caught
        """
        Extract text from multiple images in batches and format as markdown.

        Args:
            image_paths (List[str]): List of image file paths.
            batch_size (int): Number of images per batch.

        Returns:
            str: Concatenated markdown string of extracted text.
        """
        try:
            client = OpenAI()
            markdown_results = []
            prev_last_img = None
            i = 0
            batch_num = 0

            while i < len(image_paths):
                if prev_last_img is None:
                    batch = image_paths[i : i + batch_size]
                    i += batch_size
                    prompt_text = (
                        "Extract all text from the following "
                        "images and return it "
                        "formatted as markdown."
                    )
                else:
                    batch = image_paths[i : i + batch_size - 1]
                    batch = [prev_last_img] + batch
                    i += batch_size - 1
                    prompt_text = (
                        "The first image in this batch is the last "
                        "image from the previous batch, "
                        "included only for context. It should not be "
                        "repeated in the markdown output. "
                        "Extract all text from the rest of the images "
                        "and return it formatted as markdown."
                    )

                content = [{"type": "text", "text": prompt_text}]
                batch_num += 1
                print(f"[DEBUG] Processing batch {batch_num} with {len(batch)} images")

                for img_path in batch:
                    with open(img_path, "rb") as img_file:
                        img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        }
                    )

                response = client.chat.completions.create(
                    model="gpt-4.1", messages=[{"role": "user", "content": content}]
                )

                markdown = response.choices[0].message.content
                if markdown:
                    markdown = markdown.replace("```markdown", "").replace("```", "")
                    markdown_results.append(markdown)

                prev_last_img = batch[-1] if batch else None

            return "\n\n".join(markdown_results)

        except Exception as e:
            print(f"[ERROR] extract_images_to_markdown failed: {e}")
            return ""

    @staticmethod
    def supported_formats() -> List[str]:
        """
        Return the list of supported image file formats.

        Returns:
            List[str]: Supported image extensions.
        """
        return ["jpg", "jpeg", "png", "bmp", "gif"]
