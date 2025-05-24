# MosaicRX

MosaicRX is a user-friendly medicine recommendation web application. Enter your symptoms, and MosaicRX will suggest medicines using TF-IDF, cosine similarity, and SVD to rank the most relevant results!

---

## Features

- **Symptom-based search**  
  Type in one or more symptoms to retrieve a ranked list of possible medications.  
- **Filters and Sorting**  
  Filter or sort results by rating, price, or allergies.
- **Clean UI**  
  Minimalist but clean design for fast lookups.  
- **Flask-powered backend**  
  Easy to extend or swap out the recommendation engine.  

---

## Live Demo

Try it online here:  
http://4300showcase.infosci.cornell.edu:5266/

---

## Installation & Local Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/alvaroderas/MosaicRX.git
   cd MosaicRX
   ```
2. **Install dependencies**  
   ```bash
   python -m pip install -r requirements.txt
   ```
3. **Run the server**  
   ```bash
   cd backend
   flask run --host=0.0.0.0 --port=5000
   ```
4. **Open your browser**  
   Visit `http://localhost:5000/` and start searching!

---

## Usage

1. **Enter one or more symptoms** in the search box.  
2. **Submit** and MosaicRX will display a ranked list of candidate medications along with brief descriptions.  
3. **Click** one of the options on any medicine to view more details or user reviews.

---

## Contributors

- Alvaro Deras  
- Serena Inderjit  
- Dwain Anderson  
- Asher Cai  
- Eric Liu

*Final project for CS/INFO 4300: Language and Information at Cornell University*
