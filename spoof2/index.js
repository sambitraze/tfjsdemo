const videoElem = document.getElementById("videoElem");
var script = document.createElement('script');
script.src = 'https://rawgit.com/paulirish/memory-stats.js/master/bookmarklet.js';
document.head.appendChild(script);

window.onload = async function () {
    navigator.mediaDevices.getUserMedia({
        video: true
    }).then(async (stream) => {
        videoElem.srcObject = stream;
        const MODEL_URL = './model/model.json';
        const model = await tf.loadGraphModel(MODEL_URL);
        detection = setInterval(async () => {
            const tfImg = tf.browser.fromPixels(videoElem).resizeNearestNeighbor([224, 224])
                .toFloat()
                .expandDims();
            const prediction = await model.predict(tfImg);
            console.log(prediction);
            const values = prediction.dataSync();
            const arr = Array.from(values);
            console.log(arr);
        }, 3000);

    }).catch((err) => {
        alert(err);
    });
}