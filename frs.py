import cv2
import face_recognition
import dlib
import sqlite3
import numpy as np
import base64
import sys
from flask import Flask, jsonify, request
import warnings
from datetime import datetime
import json
import pprint
import os

from flask_cors import CORS







# Had to do the following too:
#   `sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y`

# Ignore deprecation warnings for cleaner output.
warnings.filterwarnings("ignore", category=DeprecationWarning)


""" DEBUGGING 

# For Debugging Only
# People for predictions!
dwayne_johnson  = ['faces/dwayne_johnson/dwayne_johnson_002.jpg', 'Dwayne Johnson']
shania_twain    = ['faces/shania_twain/shania_twain_002.jpg', 'Shania Twain']
chris_rock      = ['faces/chris_rock/chris_rock_002.jpg', 'Chris Rock']
rafael_grossi   = ['faces/rafael_grossi/rafael_grossi_002.jpg', 'Rafael Grossi']
predictions     = [dwayne_johnson, shania_twain, chris_rock, rafael_grossi]

END DEBUGGING """


detector = dlib.get_frontal_face_detector()

model_path = os.path.abspath("C:/pps-frs/shape_predictor_68_face_landmarks.dat")

# Verify the absolute path
print("Absolute path to model file:", model_path)

# Check if the file exists
if os.path.exists(model_path):
    predictor = dlib.shape_predictor(model_path)
else:
    print("Error: Model file not found.")


# Define app, ip address, and port number
app = Flask(__name__)

CORS(app)


# ip_addr =  '192.168.10.209'
# port = 4410


# Route for default request @ http://{ip_addr}:{port}/
# Requires:         None
# Returns:          message:JSON - Information message identifying the program.
# Expected Action:  Returns a fairly standard default JSON message to the request
@app.route("/")
def hello():
    return jsonify(
        message="[INFO] Connection established to the Facial Recognition System. Please use API commands."
    )


# Function to log to `all.log` and a specific file of the user's choosing
# Requires:         filename:string - the filename of the user-chosen log
# 					request:string - the request that is being processed,
#                                       use `ip_addr` (this machine's IP)  if it's just an internal method call
# 					src_ip:string - the source ip address that made the request - can be 127.0.0.1 if an internal call (e.g. auth.log)
# 					log_type:int - a number representing the type of log, 0=INFO, 1=ERROR, 2=FATAL
# 					content:string - data to be written to the file
# Returns:          return_code:int - a return code for the logging attempt, 0=FAIL, 1=SUCCESS
# Expected Action:  Open log file, write JSON data, close file
def log(filename, request, src_ip, log_type, content):
    filename = "log/" + filename

    return_code = 0  # Default = FAIL

    # Get timestamp
    date_time = datetime.now()
    timestamp = date_time.strftime("%d-%m-%Y @ %H:%M:%S")

    # Set the string for the log_type

    # log_type = 0 # info - information including authentication success and failure
    # log_type = 1 # error - errors in program execution
    # log_type = 2 # fatal - fatal errors that cause the program to exit/shutdown
    if log_type == 0:
        str_log_type = "INFO"
    elif log_type == 1:
        str_log_type = "ERROR"
    elif log_type == 2:
        str_log_type = "FATAL"
    else:
        print("[ERROR] Error writing to log file!")
        return return_code

    req = str(request)
    ip = str(src_ip)
    log_content = content

    log_entry = {
        "timestamp": timestamp,
        "request": req,
        "source_ip": ip,
        "log_type": str_log_type,
        "log_content": log_content,
    }
    log_entry_dump = json.dumps(log_entry, indent=4)

    # Write to master log
    try:
        with open("log/all.log", "a") as file:
            file.write(log_entry_dump)
            file.close()
            return_code = 1
    except:
        print("[ERROR] Error writing to log file!")

    # Write to individual log
    try:
        with open(filename, "a") as file:
            file.write(log_entry_dump)
            file.close()
            return_code = 1
    except:
        print("[ERROR] Error writing to log file!")

    pprint.pprint(log_entry)

    return return_code


# Function to read all data from the database of faces
# Requires:         None
# Returns:          db_rows:JSON - json formatted database data
# Expected Action:  Connect to db, query, return all data, log
def read_database():
    req = "internal_method_call"
    src_ip = "localhost"
    try:
        conn = sqlite3.connect("db/faces.db")
        cursor = conn.cursor()
        cursor.execute("""SELECT * FROM faces;""")
        db_rows = cursor.fetchall()
        cursor.close()
        conn.close()
        msg = "[INFO] Data read from database and returned."
        log("read_db.log", req, src_ip, 0, msg)
        return jsonify(data=db_rows)
    except:
        msg = "[ERROR] Error reading data from database."
        log("read_db.log", req, src_ip, 1, msg)


# Function to get a face encoding (facial feature mapping)
# Requires:         image_path:string - a filepath to an image of a face, e.g. 'path/to/face.jpg'
# Returns:          face_encoding:numpy.ndarray<float> - a numerical array that describes the facial features
# Expected Action:  Loads file, encodes face, returns encoding
def get_face_encoding(image_path):
    # Load image
    image = face_recognition.load_image_file(image_path)
    # Encode image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    face_encoding = face_encodings[0]

    return face_encoding


# Function for loading all encoded faces from the databases
# Requires:         None
# Returns:          face_encodings:list - a list of numpy.ndarrays in string format
# Expected Action:  Connect to db, select all encoding, return all encodings in list
def load_db_face_encodings():
    req = "internal_method_call"
    src_ip = "localhost"
    try:
        conn = sqlite3.connect("db/faces.db")
        cursor = conn.cursor()
        cursor.execute("SELECT employee_id, name, encoding FROM faces")
        face_encodings = cursor.fetchall()
        cursor.close()
        conn.close()
        msg = "[INFO] Loaded face encondings from database."
        log("read_db.log", req, src_ip, 0, msg)
        return face_encodings
    except:
        msg = "[FAIL] Failed to load face encodings from database."
        msg = msg + " " + "[EXIT] Exiting program now."
        log("read_db.log", req, src_ip, 2, msg)
        sys.exit(0)


# Function to compare two face encodings and produce a simlarity score
# Requires:         known_face:numpy.ndarray<float> - a face encoding that is linked to/stored in the database (i.e. a known face)
#                   input_face_encoding:numpy.ndarray<float> - a face encoding from an input image
# Returns:          similarity_score:float - a score between (roughly between -1 and 1) that reveals how close the faces are.
# Expected Action:  Calculates the face distance between the two arrays of face encodings, returns a similarity score
def compare_faces(known_face, input_face_encoding):
    # Calculate the Euclidean distance between two face encodings
    face_distance = face_recognition.face_distance([known_face], input_face_encoding)
    # Convert the distance to a similarity score (1 - distance)
    similarity_score = 1 - face_distance[0]
    return similarity_score


# Function to find the best match for a given image by cross-referencing the image to all known faces in the database
# Requires:         input_image_path:string - a filepath to an image of a face
# Returns:          best_match_id:int - the ID of the best matching face in the database
#                   best_match_score:flaot - the similarity score for the best match in the database - this can be used to define thresholds.
# Expected Action:  Encode input face, get all encoded faces from db, compare input face to all in the db and get scores for similarity, return best matching face id and score
def match_image(input_image_path):
    # Get encoding of the input image
    input_face_encoding = get_face_encoding(input_image_path)  # input_face_encodings[0]

    # Load the face encodings from the database
    db_face_encodings = load_db_face_encodings()

    best_match_id = None
    best_match_score = 0

    # Iterate over each face encoding in the database
    for employee_id, face_name, face_encoding in db_face_encodings:
        known_face = np.fromstring(face_encoding)

        similarity_score = compare_faces(known_face, input_face_encoding)
        similarity_score = similarity_score * 2
        if similarity_score >= 1.0:
            similarity_score = 0.99999
        if similarity_score <= -1.0:
            similarity_score = -0.99999
        if similarity_score > best_match_score:
            best_match_score = similarity_score
            best_match_id = employee_id

        print(f"\t\t[INFO] Similarity score for {face_name}: {similarity_score}")

    return best_match_id, best_match_score


# Route for requesting a face match lookup
# Requires:         None - although an image must be sent via POST request with the key of `image`
# Returns:          JSON(data:list) - a list containing the best match ID of the face, and the similarity score
# Expected Action:  Take input image and call the match() function, return the results as JSON
@app.route("/match/", methods=["POST"])
def process_request():
    req = str(request)
    src_ip = str(request.remote_addr)

    data = request.json
    image_req = base64.b64decode(data["image"])

    with open("tmp_face.jpg", "wb") as fh:
        fh.write(image_req)

    if image_req != None:
        try:
            best_match_id, best_match_score = match_image("tmp_face.jpg")
            data = [best_match_id, best_match_score]
            msg = f"[INFO] Best match for image provided: Employee {best_match_id}. Confidence Score {best_match_score}."
            log("face_id.log", req, src_ip, 0, msg)
            return jsonify(data=data)
        except:
            msg = "[FAIL] Error: Match not found for image provided."
            log("face_id.log", req, src_ip, 0, msg)
            return jsonify(message=msg)
        


def format_recent_logs(input_file, output_file, limit):
    # Read data from the input file
    with open(input_file, 'r') as file:
        log_entries = file.read().strip()
 
    # Modify the delimiter to properly split JSON objects
    json_objects = []
    start = 0
    for end in range(len(log_entries)):
        if log_entries[end] == '}':
            json_objects.append(json.loads(log_entries[start:end + 1]))
            start = end + 1

    # Limit the number of logs to the specified 'limit' from the end
    json_objects = json_objects[::-1][:limit]
 
    # Write formatted data into the output JSON file
    with open(output_file, 'w') as file:
        file.write(json.dumps(json_objects, indent=20))

# Example usage:
input_file_path = 'log/all.log'
output_file_path = 'formatted_all.log.json'
format_recent_logs(input_file_path, output_file_path, limit=20)



@app.route('/api/get_data', methods=['GET'])
def get_data():
    # Read your JSON file and return it
    # For simplicity, let's assume you have a file named data.json in the same directory
    with open('formatted_all.log.json', 'r') as file:
        data = file.read()
    return data



# Main Program!
if __name__ == "__main__":
    # f_e = get_face_encoding('input_face.jpg')
    # print(f_e)
    # best_match_id, best_match_score = match_image('input_face.jpg')
    # data = [best_match_id, best_match_score]
    # msg = f"[INFO] Best match for image provided: Employee {best_match_id}. Confidence Score {best_match_score}."
    # print(msg)

    flask_msg = "[FLASK] Starting Flask Server..."
    log("system_state.log", "__main__", "localhost", 0, flask_msg)
    try:
        app.run(debug=True, port=7001)
    except:
        msg = "[ERROR] Fault encountered while attempting to run the Flask API server. [EXIT] The system will now exit."
        log("system_state.log", "__main__", "localhost", 2, flask_msg)
        sys.exit(0)

    """ DEBUGGING 
    # For Debugging Only
    for person in predictions:
        print(f'\n[INFO] Predicting {person[1]}:')
        image_path = person[0]
        best_match_id, best_match_score = match_image(image_path)
        try:
            conn = sqlite3.connect('db/faces.db')
            cursor = conn.cursor()
            cursor.execute('SELECT name FROM faces WHERE id = (?);', (best_match_id,))
            matched_person = cursor.fetchone()
            matched_person = matched_person[0]
            print(f'[MATCH] Matched image of {person[1]} with {matched_person}\'s image in database with a confidence score of {best_match_score}')
            cursor.close()
            conn.close()
        except:
            print('[FAIL] Best Match ID not returned...')
    END DEBUGGING """
