var video = document.querySelector("#videoElement");

function start(e) {
    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then( async function (stream) {
                video.srcObject = stream;
                const model = await downloadModel();             
                detection = setInterval(() => {
                    yolo(video, model).then(predictions => {
                        console.log(predictions);
                        // var canvas = document.querySelector("#videoCanvas");
                        // var ctx = canvas.getContext('2d');
                        // canvas.width = video.videoWidth;
                        // canvas.height = video.videoHeight;
                        // ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        // predictions.forEach(prediction => {
                            // const {
                            //     top, left, bottom, right, classProb, className,
                            // } = prediction;
                            // ctx.strokeStyle = "red";
                            // ctx.fillStyle = "red";
                            // ctx.font = "20pt sans-serif";
                            // ctx.fillText(className + " (" + (classProb * 100).toFixed(4) + ")", left, top);
                            // ctx.strokeRect(left, top, right - left, bottom - top,);
                            // if(prediction.class === "cell phone"){
                            //     ctx.strokeStyle = "red";
                            //     ctx.fillStyle = "red";
                            //     ctx.font = "20pt sans-serif";
                            //     ctx.fillText(className + " (" + (classProb * 100).toFixed(4) + ")", left, top);
                            //     ctx.strokeRect(left, top, right-left, bottom-top,);
                            // }
                            // if(prediction.class === "book"){
                            //     const x = prediction.bbox[0];
                            //     const y = prediction.bbox[1];
                            //     const width = prediction.bbox[2];
                            //     const height = prediction.bbox[3];
                            //     ctx.strokeStyle = "red";
                            //     ctx.fillStyle = "red";
                            //     ctx.font = "20pt sans-serif";
                            //     ctx.fillText(prediction.class + " (" + (prediction.score * 100).toFixed(4) + ")", x, y);
                            //     ctx.strokeRect(x, y, width, height);
                            // }
                        // });
                    });
                }, 100);
            })
            .catch(function (err0r) {
                console.log(err0r);
                console.log("Something went wrong!");
            });
    }
}

function stop(e) {
    var stream = video.srcObject;
    var tracks = stream.getTracks();

    for (var i = 0; i < tracks.length; i++) {
        var track = tracks[i];
        track.stop();
        clearTimeout(detection);
    }

    video.srcObject = null;
}
// var video = document.querySelector("#webCamera");
// video.onplay = function() {

// };
