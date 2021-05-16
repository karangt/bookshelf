# Real Bookshelf to Virtual Bookshelf 📚
The idea is to take an image of a real life bookshelf and generate a digital list of books, along with a Goodreads link for each of them.

## Project Structure

- `Final_Project_Bookshelf.ipynb`: This is the main notebook where the experimentation was done and the system was implemented. This was eventually refactored to shift all the core functions to the individual .py file.
- `app.ipynb`: This notebook is *the* interactive application hosted at http://bookshelf.karangupta.net.
- `east.py`: Contains functions for detecting text in images using the EAST Text Detection model.
- `gocr.py`: Contains functions for doing OCR using Google Cloud Vision API and for generating Google and GoodReads links for the books.
- `segment.py`: Contains functions for segmenting an image into segments corresponding to individual books.
- `utils.py`: Contains utility functions used across the project.
- `google_app_creds.default.json`: Contains the credentials for calling the Google Cloud Vision API. For more info on getting the credentials, please check this [link](https://cloud.google.com/vision/docs/setup). This file should be renamed to `google_app_creds.json` before deployment.
- `requirements.txt`: File used to install project dependencies.
- `./images`: Directory that contains sample images.
- `./experiments`: Directory that contains notebooks with various experiments done over the course of the project (may not work now).

## Running The Project
1. Make sure Python3 is installed.
2. Install Opencv on a Debian system using: `sudo apt install python3-opencv`
3. Install the dependencies by running: `pip3 install -r requirements.txt`
4. Launch Jupyter notebook server and run the `Final_Project_Bookshelf.ipynb` notebook as usual.
5. Launch the application by running: `voila app.ipynb`












