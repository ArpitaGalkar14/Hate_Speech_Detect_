const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");
const { PythonShell } = require("python-shell");

const app = express();
app.use(bodyParser.json());
app.use(cors());

app.post("/predict", (req, res) => {
    const tweet = req.body.tweet;

    let options = {
        mode: "text",
        pythonOptions: ["-u"],
        args: [tweet]
    };

    PythonShell.run("predict.py", options, function (err, results) {
        if (err) throw err;
        res.send(JSON.parse(results[0]));
    });
});

app.listen(5000, () => {
    console.log("Node server running at http://localhost:5000");
});
