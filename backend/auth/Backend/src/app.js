import express from 'express';
import cors from "cors";
import authRoutes from './routes/auth.routes.js';


const app = express();
app.use(express.json());

app.use(
    cors({
        origin: process.env.CORS_ORIGIN,
        credentials: true
    })
)

app.use('/api/auth', authRoutes);



// app.use(cors());



app.get('/', (req, res) => {
  res.send('Backend server started!');
});

export default app;