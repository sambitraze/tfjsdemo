const videoElem = document.getElementById("videoElem");
const canvasRef = document.getElementById("canvasRef");
var script = document.createElement('script');
script.src = 'https://rawgit.com/paulirish/memory-stats.js/master/bookmarklet.js';
document.head.appendChild(script);

const labelMap = {
    1: { name: 'Book', color: 'red' },
    2: { name: 'Phone', color: 'yellow' },
}

// Define a drawing function
drawRect = (boxes, classes, scores, threshold, imgWidth, imgHeight, ctx) => {
    ctx.drawImage(videoElem, 0, 0, imgWidth, imgHeight);
    for (let i = 0; i <= boxes.length; i++) {
        if (boxes[i] && classes[i] && scores[i] > threshold) {
            console.log("yes");
            const [y, x, height, width] = boxes[i]
            const text = classes[i]
            // Set styling
            ctx.strokeStyle = labelMap[text]['color']
            ctx.lineWidth = 2
            ctx.fillStyle = 'white'
            ctx.font = '30px Arial'

            // DRAW!!
            ctx.beginPath()
            ctx.fillText(labelMap[text]['name'] + ' - ' + Math.round(scores[i] * 100) / 100, x * imgWidth, y * imgHeight - 10)
            ctx.rect(x * imgWidth, y * imgHeight, width * imgWidth / 2, height * imgHeight / 2);
            ctx.stroke()
        }
    }
}

window.onload = async function () {
    navigator.mediaDevices.getUserMedia({
        video: true
    }).then(async (stream) => {
        videoElem.srcObject = stream;
        const MODEL_URL = './model/model.json';
        const model = await tf.loadGraphModel(MODEL_URL);
        detection = setInterval(async () => {
            detect(model);
        }, 100);

    }).catch((err) => {
        alert(err);
    });
}

const detect = async (net) => {    
    const videoWidth = videoElem.videoWidth;
    const videoHeight = videoElem.videoHeight;
    canvasRef.width = videoElem.videoWidth;
    canvasRef.height = videoElem.videoHeight;

    // 4. TODO - Make Detections
    const img = tf.browser.fromPixels(videoElem);
    const resized = tf.image.resizeBilinear(img, [640, 640]);
    const casted = resized.cast("int32");
    const expanded = casted.expandDims(0);
    const obj = await net.executeAsync(expanded);

    const boxes = await obj[4].array(); //
    const classes = await obj[6].array(); //Classes  
    const scores = await obj[0].array(); // 

    // Draw mesh
    const ctx = canvasRef.getContext("2d");

    // 5. TODO - Update drawing utility
    // drawSomething(obj, ctx)
    requestAnimationFrame(() => {
        drawRect(
            boxes[0],
            classes[0],
            scores[0],
            0.5,
            videoWidth,
            videoHeight,
            ctx
        );
    });

    tf.dispose(img);
    tf.dispose(resized);
    tf.dispose(casted);
    tf.dispose(expanded);
    tf.dispose(obj);
};