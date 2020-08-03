let mobilenet;
let model;
const webcam = new Webcam(document.getElementById("wc"));

var rocks = 0;
var papers = 0;
var scissors = 0;

var dataset = new RPSDataset();
let isPredicting = false;

async function init() {
    await webcam.setup();
    mobilenet = await loadMobilenet();
    tf.tidy(() => mobilenet.predict(webcam.capture()));
}

async function loadMobilenet() {
    const mobilenet = await tf.loadLayersModel(
        "./model.json"
    );
    
    const layer = mobilenet.getLayer("conv_pw_13_relu");
    console.log("Mobilenet loaded");
    return tf.model({
        inputs: mobilenet.inputs,
        outputs: layer.output
    });
}

function handleBtn(elem) {
    switch(elem.id) {
        case "0":
            rocks++;
            document.getElementById("rock").innerText = "Pranjal: " + rocks;
            break;
        case "1":
            papers++;
            document.getElementById("paper").innerText = "Samman: " + papers;
            break;
        case "2":
            scissors++;
            document.getElementById("scissor").innerText = "Shantonu: " + scissors;
            break;
    }

    let label = parseInt(elem.id);
    const img = webcam.capture();
    dataset.addExample(mobilenet.predict(img), label);
}

async function train() {
    dataset.ys = null;
    dataset.encodeLabels(3);

    model = tf.sequential({
        layers: [
            tf.layers.flatten({
                inputShape: [7, 7, 256]
            }),
            tf.layers.dense({
                units: 100,
                activation: "relu"
            }),
            tf.layers.dense({
                units: 3,
                activation: "softmax"
            })
        ]
    });

    const optimizer = tf.train.adam(0.0001);
    model.compile({
        optimizer: optimizer,
        loss: "categoricalCrossentropy"
    })

    let loss = 0;
    model.fit(dataset.xs, dataset.ys, {
        epochs: 10,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                loss = logs.loss.toFixed(5);
                console.log("Loss: " + loss);
            }
        }
    });
}

async function predict() {
    while (isPredicting) {
        const predictedClass = tf.tidy(() => {
            const img = webcam.capture();
            const activation = mobilenet.predict(img);
            const prediction = model.predict(activation);
            return prediction.as1D().argMax();
        });

        const classId = (await predictedClass.data())[0];
        var predictionText = "";
        switch(classId){
            case 0:
                predictionText = "I see Pranjal";
                break;
            case 1:
                predictionText = "I see Samman";
                break;
            case 2:
                predictionText = "I see Shantonu";
                break;
        }
        document.getElementById("pred").innerText = predictionText;
                
        predictedClass.dispose();
        await tf.nextFrame();
    }
}

function doTraining(){
	train();
}

function trigger(flag) {
    isPredicting = flag;
    predict();
}

init();