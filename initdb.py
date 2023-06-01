import sqlite3
from frs import log
import sys
import cv2
import face_recognition


# Function for initialising database with faces.
# Requires:         None
# Returns:          None
# Expected Action:  Connect to db, create table and encode and insert faces from local files, log
def setup_db():
    global people
    req = "internal_method_call"
    src_ip = "localhost"
    msg = "[START] Initialising database."
    try:
        conn = sqlite3.connect("db/faces.db")
        cursor = conn.cursor()
        cursor.execute("""DROP TABLE IF EXISTS faces;""")
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS faces
                    (employee_id INTEGER PRIMARY KEY,
                    name text UNIQUE NOT NULL,
                    encoding TEXT NOT NULL)"""
        )
        msg = msg + "" + "[INFO] Table created successfully."

        cursor.close()
        conn.close()
    except:
        msg = "[FAIL] Unable to create table."
        msg = msg + " " + "[EXIT] Exiting program now."
        log("write_db.log", req, src_ip, 2, msg)
        sys.exit(0)

    # People for inserting into database!
    dwayne_johnson = [
        101,
        "Dwayne Johnson",
        "faces/dwayne_johnson/dwayne_johnson_001.jpg",
    ]
    shania_twain = [102, "Shania Twain", "faces/shania_twain/shania_twain_001.jpg"]
    chris_rock = [103, "Chris Rock", "faces/chris_rock/chris_rock_001.jpg"]
    rafael_grossi = [104, "Rafael Grossi", "faces/rafael_grossi/rafael_grossi_001.jpg"]
    people = [dwayne_johnson, shania_twain, chris_rock, rafael_grossi]

    msg = msg + " " + "[INFO] Adding faces to database..."
    for person in people:
        # Encode the image from the path (2) and store the encoding, employee_id (0) and the name (1) in the database
        encode_and_store_face(person[0], person[1], person[2])

    log("write_db.log", req, src_ip, 0, msg)


# Function to encode face and store it in a database
# Requires:         image_path:string - a filepath to an image of a face, e.g. 'path/to/face.jpg'
#                   name:string a string
#                   employee_id:int - the employee_id that matches employee_id the Access Control System (separate machine)
# Returns:          None
# Expected Action:  Connect to db, encode face, convert to string, store in db along with other inputs, log
def encode_and_store_face(employee_id, name, image_path):
    req = "internal_method_call"
    src_ip = "localhost"
    try:
        conn = sqlite3.connect("db/faces.db")
        cursor = conn.cursor()

        # Load the image
        image = face_recognition.load_image_file(image_path)
        # Encode image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_encoding = face_encodings[0]

        # Encoding the face in base64
        face_string = face_encoding.tostring()

        # Store the face encoding in the database
        cursor.execute(
            "INSERT INTO faces (employee_id, name, encoding) VALUES (?, ?, ?)",
            (employee_id, name, face_string),
        )
        conn.commit()
        msg = f"[INFO] Inserted the face of [{employee_id}] {name} into the database.\n\t{image_path}"
        log("write_db.log", req, src_ip, 0, msg)
        cursor.close()
        conn.close()
    except:
        msg = f"[ERROR] Failed to insert the face of [{employee_id}] {name} into the database.\n\t{image_path}"
        msg = msg + " " + "[EXIT] Exiting program now."
        log("write_db.log", req, src_ip, 2, msg)
        sys.exit(0)


# Main Program!
if __name__ == "__main__":
    setup_db()
