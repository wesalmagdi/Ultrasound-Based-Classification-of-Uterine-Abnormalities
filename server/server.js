const express = require('express');
const cors = require('cors'); // <--- 1. Import this
const app = express();

app.use(cors()); // <--- 2. Enable this BEFORE your routes
app.use(express.json());

app.post('/api/predict', (req, res) => {
   console.log("Request received from React!");
   // your prediction logic here
});