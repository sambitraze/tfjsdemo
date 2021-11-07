const imageScaleFactor = 0.50;
const flipHorizontal = true;
const outputStride = 16;

window.onload = function () {
    navigator.mediaDevices.getUserMedia({
        video: true
    }).then(async (stream) => {
        const video = document.getElementById('video');
        video.srcObject = stream;
        const net = await posenet.load();
        video.play();
        detection = setInterval(async () => {
            const canvas = document.getElementById('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            const pose = await net.estimateSinglePose(video, imageScaleFactor, flipHorizontal, outputStride);
            let nsx = pose.keypoints[0].position.x;
            let nsy = pose.keypoints[0].position.y;
            let lex = pose.keypoints[1].position.x;
            let ley = pose.keypoints[1].position.y;
            let rex = pose.keypoints[2].position.x;
            let rey = pose.keypoints[2].position.y;
            const distToLeftEyeX = Math.abs(lex - nsx);
            const distToRightEyeX = Math.abs(rex - nsx);

            if ((distToRightEyeX - distToLeftEyeX) > 10) {
                console.log("Looking Left");
            } else if ((distToLeftEyeX - distToRightEyeX) > 10) {
                console.log("Looking Right");
            } else {
                console.log("Looking Straight");
            }
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        }, 500);

    });
}
