import easyocr

reader = easyocr.Reader(['en'], verbose=False)

def ocr(frame):
    results = reader.readtext(frame)

    # Sort roughly top → bottom
    results.sort(key=lambda x: x[0][0][1])

    text = []

    for bbox, detected_text, confidence in results:
        if confidence > 0.4:
            text.append(detected_text)

    return " ".join(text)