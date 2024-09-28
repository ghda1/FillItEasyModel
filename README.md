### FiilItEasy Model:


This project develops a model for converting an image to text using machine learning (ML) techniques and optical character recognition (OCR).
The developed model will be capable of converting an image that includes the students’ marks into textual data.
Then, the system will fill the extracted data into an excel table. There are various tasks in the pipeline to develop such model, firstly, features will be extracted once the instructor inputs the file or image. 
The feature extraction step converts each character image into a set of representative numerical features.
Then, the layout analysis and table structure detection stages will be applied to analyze the visual structure of images, define table boundaries, and identify rows and columns within the table.
After that, the OCR pretrained models, such as PaddleOCR and EasyOCR, is applied to recognize text from tables and performs post-processing of the table’s data. Finally, the evaluation process will be applied to ensure that the used model provides best performance.
Results shows that hybrid model which combined both PaddelOCR and EasyOCR models provides the best performance with an accuracy of 94.2%.
