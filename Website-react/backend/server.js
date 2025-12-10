const express = require('express');
const bodyParser = require('body-parser');
const sqlite3 = require('sqlite3').verbose();
const session = require('express-session');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const nodemailer = require('nodemailer');
const { PythonShell } = require('python-shell');
const cors = require('cors');

const app = express();
const db = new sqlite3.Database('users.db');

// ---------------- MIDDLEWARE ----------------
app.use(cors({
  origin: "http://localhost:3000", // React frontend
  credentials: true
}));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// ---------------- SESSION ----------------
app.use(session({
  secret: 'mySecretKey',
  resave: false,
  saveUninitialized: false,
  cookie: {
    httpOnly: true,
    secure: false,      // must be false for localhost
    sameSite: 'lax'     // allows cross-origin cookies
  }
}));

// ---------------- STATIC FOLDERS ----------------
app.use('/videos', express.static(path.join(__dirname, 'videos')));
app.use('/models', express.static(path.join(__dirname, 'models')));
app.use('/answers', express.static(path.join(__dirname, 'answers')));

// Ensure answers folder exists
const answersDir = path.join(__dirname, 'answers');
if (!fs.existsSync(answersDir)) fs.mkdirSync(answersDir);

// ---------------- MULTER ----------------
const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, 'answers/'),
  filename: (req, file, cb) => {
    const { username, sessionTime } = req.session;
    const q = req.body.question || 'unknown';
    cb(null, `${username}_${q}_${sessionTime}.webm`);
  }
});
const upload = multer({ storage });

// ---------------- EMAIL ----------------
const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: 'aditideo624@gmail.com',
    pass: 'byauhfzxgjvjbyeg',
  }
});
transporter.verify(err => {
  if (err) console.error('❌ Nodemailer error:', err);
  else console.log('✅ Email server ready');
});

// ---------------- DATABASE SETUP ----------------
db.serialize(() => {
  db.run(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE,
      password TEXT,
      email TEXT
    )
  `);

  db.run(`
    CREATE TABLE IF NOT EXISTS answers (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT,
      question INTEGER,
      answer TEXT,
      timestamp INTEGER,
      UNIQUE(username, question, timestamp)
    )
  `);
});

// ---------------- ROUTES ----------------

// Register
app.post('/register', (req, res) => {
  const { username, password, email } = req.body;
  db.run(
    'INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
    [username, password, email],
    err => {
      if (err) return res.send('User already exists.');
      res.send('Registered successfully.');
    }
  );
});

// Login
app.post('/login', (req, res) => {
  const { username, password } = req.body;
  db.get(
    'SELECT * FROM users WHERE username = ? AND password = ?',
    [username, password],
    (err, row) => {
      if (row) {
        req.session.loggedIn = true;
        req.session.username = username;
        req.session.sessionTime = Date.now();
        console.log('Session after login:', req.session); // DEBUG
        res.json({ success: true });
      } else {
        res.json({ success: false });
      }
    }
  );
});

// Save TEXT answer
app.post('/save-answer', (req, res) => {
  if (!req.session.loggedIn) return res.status(401).send('Unauthorized');

  const { question, answer } = req.body;
  const { username, sessionTime } = req.session;

  db.run(
    `
    INSERT INTO answers (username, question, answer, timestamp)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(username, question, timestamp)
    DO UPDATE SET answer = excluded.answer
    `,
    [username, question, answer, sessionTime],
    err => {
      if (err) return res.status(500).send('DB error');
      res.send('Saved');
    }
  );
});

// Upload VIDEO answer
app.post('/upload-answer', upload.single('video'), (req, res) => {
  if (!req.session.loggedIn) return res.status(401).send('Unauthorized');
  res.send('Video uploaded');
});

// Upload transcript
app.post('/upload-transcript', (req, res) => {
  if (!req.session.loggedIn) return res.status(401).send('Unauthorized');

  const { question, transcript } = req.body;
  const { username, sessionTime } = req.session;

  db.run(
    `
    INSERT INTO answers (username, question, answer, timestamp)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(username, question, timestamp)
    DO UPDATE SET answer = excluded.answer
    `,
    [username, question, transcript, sessionTime],
    err => {
      if (err) return res.status(500).send('Failed to save transcript');
      res.send('Transcript saved');
    }
  );
});

// Choose domain
app.post('/choose-domain', (req, res) => {
  if (!req.session.loggedIn) return res.status(401).send('Unauthorized');

  req.session.domain = req.body.domain;
  res.json({ success: true });
});

// ---------------- GET VIDEOS ----------------
app.get('/get-videos', (req, res) => {
  if (!req.session.loggedIn) return res.status(401).send('Unauthorized');

  const domain = req.query.domain || req.session.domain;
  if (!domain) return res.status(400).send('Domain not specified');

  const domainFolder = path.join(__dirname, 'videos', domain);
  if (!fs.existsSync(domainFolder)) {
    console.log("❌ Folder not found:", domainFolder);
    return res.json([]);
  }

  const files = fs
    .readdirSync(domainFolder)
    .filter(f => f.match(/\.(mp4|webm|mov|mkv|avi)$/i))
    .map(f => `/videos/${domain}/${encodeURIComponent(f)}`);

  console.log("⭐ Sending files:", files);
  return res.json(files);
});

// ---------------- PYTHON ANALYSIS ----------------
const scriptsPath = path.join(__dirname, 'Scripts');

app.get('/analyze/eye', (req, res) => {
  PythonShell.run('eye_detection.py', { pythonPath: 'python', scriptPath: scriptsPath }, (err, results) => {
    if (err) return res.status(500).send('Eye detection failed');
    res.json({ success: true, output: results });
  });
});

app.get('/analyze/fer', (req, res) => {
  PythonShell.run('emotion_detection.py', { pythonPath: 'python', scriptPath: scriptsPath }, (err, results) => {
    if (err) return res.status(500).send('FER failed');
    res.json({ success: true, output: results });
  });
});

app.get('/analyze/posture', (req, res) => {
  PythonShell.run('body_posture.py', { pythonPath: 'python', scriptPath: scriptsPath }, (err, results) => {
    if (err) return res.status(500).send('Posture failed');
    res.json({ success: true, output: results });
  });
});

// ---------------- START SERVER ----------------
app.listen(5000, () => console.log('🚀 Backend running on http://localhost:5000'));
