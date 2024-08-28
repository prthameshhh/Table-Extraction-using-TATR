# Table Transformer for Table Detection and Structure Recognition and table extraction

This project implements a pipeline for detecting tables in images and recognizing their structure using the Table Transformer (TATR) model. It includes table detection, structure recognition, and OCR capabilities.

## Description

This tool utilizes state-of-the-art deep learning models to automate the process of extracting tabular data from images. It employs the Table Transformer (TATR) model for both table detection and structure recognition, followed by Optical Character Recognition (OCR) to extract the text content. This pipeline is particularly useful for digitizing printed documents, analyzing scanned reports, or processing image-based datasets containing tabular information.

The workflow consists of three main steps:
1. Table Detection: Identifies and localizes tables within the input image.
2. Structure Recognition: Analyzes the structure of detected tables, identifying rows, columns, and individual cells.
3. OCR: Extracts the text content from each cell of the recognized table structure.

The results are visualized at each step and the final extracted data is saved in a convenient CSV format for further analysis or integration into other workflows.

## Features

- Table detection in images
- Table structure recognition
- Optical Character Recognition (OCR) for table contents
- Visualization of detected tables and recognized structures
- Export of table contents to CSV

## Requirements

- Python 3.7+
- CUDA-capable GPU (recommended for faster processing)

For a complete list of required packages, see `requirements.txt`.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/table-transformer-project.git
   cd table-transformer-project
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your input image(s) containing tables

2. Update the `image_path` and `output_dir` variables in the script:
   ```python
   image_path = "/path/to/your/image.png"  # Replace with your image path
   output_dir = "/path/to/output/directory"  # Replace with your desired output directory
   ```

3. Run the script:
   ```
   python table_transformer_script.py
   ```

4. Check the output directory for results:
   - `detected_tables.jpg`: Visualization of detected tables
   - `table_1.jpg`, `table_2.jpg`, etc.: Cropped images of individual tables
   - `table_structure.jpg`: Visualization of recognized table structure
   - `output.csv`: Extracted table contents in CSV format

## References

- [Table Transformer (TATR)](https://github.com/microsoft/table-transformer)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## License

This project is licensed under the Apache License  2.O

## Acknowledgments

- Microsoft for the Table Transformer model
- The EasyOCR team for their OCR engine
- Hugging Face for their Transformers library

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
