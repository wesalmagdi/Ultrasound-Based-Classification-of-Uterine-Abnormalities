import React, { useState } from 'react';
import axios from 'axios';

function App() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [data, setData] = useState({ inf: 0, misc: 0 });
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const handleImage = (e) => {
        const file = e.target.files[0];
        setImage(file);
        setPreview(URL.createObjectURL(file));
    };

    const handleUpload = async () => {
        setLoading(true);
        const formData = new FormData();
        formData.append('image', image);
        formData.append('inf', data.inf);
        formData.append('misc', data.misc);

        try {
            const res = await axios.post('http://localhost:5000/api/predict', formData);
            setResult(res.data);
        } catch (err) {
            alert("Error connecting to AI Server");
        }
        setLoading(false);
    };

    return (
        <div style={{ padding: '40px', fontFamily: 'sans-serif', maxWidth: '600px', margin: 'auto' }}>
            <h2>🩺 Uterine Abnormality Detection</h2>
            <div style={{ marginBottom: '20px' }}>
                <label>Unexplained Infertility?</label>
                <select onChange={e => setData({...data, inf: e.target.value})}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div style={{ marginBottom: '20px' }}>
                <label>Previous Miscarriages:</label>
                <input type="number" onChange={e => setData({...data, misc: e.target.value})} />
            </div>
            <input type="file" onChange={handleImage} />
            {preview && <img src={preview} alt="Preview" style={{ width: '100%', marginTop: '10px' }} />}
            
            <button 
                onClick={handleUpload} 
                disabled={!image || loading}
                style={{ width: '100%', padding: '10px', marginTop: '20px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '5px' }}
            >
                {loading ? "Analyzing..." : "Run AI Diagnosis"}
            </button>

            {result && (
                <div style={{ marginTop: '20px', padding: '15px', backgroundColor: result.prediction === 1 ? '#f8d7da' : '#d4edda', borderRadius: '5px' }}>
                    <h3>Result: {result.label}</h3>
                </div>
            )}
        </div>
    );
}

export default App;