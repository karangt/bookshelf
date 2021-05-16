# This file contains functions that use Google Vision API to do OCR
# References:
#   Text Detection: https://cloud.google.com/vision/docs/ocr
#   Quick start: https://cloud.google.com/vision/docs/quickstart-client-libraries

import os
import cv2
from google.cloud import vision
import requests

gcreds = "google_app_creds.json"

def getGClient():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getcwd()+"/"+gcreds
    return vision.ImageAnnotatorClient()

def detectText(client, img):
    """
    Detects text in the file.
    From: https://cloud.google.com/vision/docs/ocr
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getcwd()+"/"+gcreds

    # Convert img numpy array to binary format
    content = cv2.imencode('.jpg', img)[1].tostring()

    image = vision.Image(content=content)
    # Call the text detection API
    response = client.text_detection(image=image)

    texts = response.text_annotations
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


    return  texts[0].description if len(texts)>0 else ""


def getGBooksQueryText(txt):
    '''
    Return the query for the Google Books search API
    Sample: https://www.googleapis.com/books/v1/volumes?q=search+terms
    '''
    t = txt \
        .replace('\n', ' ')\
        .replace('-', '')\
        .replace('&', '')\
        .replace('+', '')\
        .replace('  ', ' ')\
        .strip()\
        .split(' ')

    return "+".join(t)

def getGBooksInfo(txt):
    '''
    Returns the link to the Google Books page
    '''
    gbooks_url = 'https://www.googleapis.com/books/v1/volumes?q={}'
    google_url = 'https://www.google.com/search?q={}'
    goodreads_url = 'https://www.goodreads.com/search?q={}&search_type=books&search%5Bfield%5D=on'

    # Generate query
    q = getGBooksQueryText(txt)

    # Get the link to the book
    resp = requests.get(gbooks_url.format(q))
    resp_json = resp.json()


    # Create book info
    if "items" in resp_json and len(resp_json["items"])>0:
        # Return the link from the first search result
        volInfo = resp_json["items"][0]["volumeInfo"]

        # Get GoodReads search link
        author = volInfo["authors"][0] if "authors" in volInfo else "[unknown]"
        title_author = (volInfo["title"] + " " + author).replace(" ","+")
        s = requests.Session()
        gr = requests.Request('GET', goodreads_url.format(title_author)).prepare()


        return {"query":txt.replace("+"," "),\
                "title":volInfo["title"],\
                "author":author,\
                "google_link":volInfo["infoLink"],\
                "goodreads_link":gr.url}
    else:
        # Return a Google Search URL
        s = requests.Session()
        p = requests.Request('GET', google_url.format(q)).prepare()

        # Get GoodReads search link
        s = requests.Session()
        gr = requests.Request('GET', goodreads_url.format(q)).prepare()

        #print("Could not find a link for:",q)
        #print(resp_json)
        return {"query":txt.replace("+"," "),\
                "title":"[{}]".format(q.replace("+"," ")),\
                "author":"[unknown]",\
                "google_link":p.url,\
                "goodreads_link":gr.url}


def getAllBookLinks(bookImages):
    gclient = getGClient()

    # Get OCR test for all books
    bookTextList = []
    for bk in bookImages:
        text = detectText(gclient, bk)
        bookTextList.append(text)

    # Get name and links for all books
    bookInfoList = []
    for txt in bookTextList:
        info = getGBooksInfo(txt)
        bookInfoList.append(info)

    return bookInfoList

