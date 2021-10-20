const imageSize = 416;
async function main() {
    try {
        
        const options = createMySessionOptions();
        const session = await ort.InferenceSession.create('./yolov4.onnx', options);
        var image = document.getElementById("imageSource");
        var canvas = document.getElementById("myCanvas");
        // document.body.appendChild(canvas);
        canvas.width = 416;
        canvas.height = 416;
        var context = canvas.getContext("2d");
        context.drawImage(image, 0, 0, image.width, image.height, 0, 0, 416, 416);

        const preprocessedData = preProcess(context);

        const feeds = { input: preprocessedData };

        // feed inputs and run
        const results = await session.run(feeds);

        // read from results
        // const dataC = results.c.data;
        // document.write(`data of result tensor 'c': ${results.boxes.data} \n ${results.confs.data}`);
        console.log(results);

    } catch (e) {
        document.write(`failed to inference ONNX model: ${e}.`);
    }
}

// main();


function preProcess(ctx) {
    var imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    var data = imageData.data, width = imageData.width, height = imageData.height;
    var dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    var dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);
    ndarray.ops.assign(dataProcessedTensor.pick(0, 0, null, null), dataTensor.pick(null, null, 2));
    ndarray.ops.assign(dataProcessedTensor.pick(0, 1, null, null), dataTensor.pick(null, null, 1));
    ndarray.ops.assign(dataProcessedTensor.pick(0, 2, null, null), dataTensor.pick(null, null, 0));
    ndarray.ops.divseq(dataProcessedTensor, 255);
    ndarray.ops.subseq(dataProcessedTensor.pick(0, 0, null, null), 0.485);
    ndarray.ops.subseq(dataProcessedTensor.pick(0, 1, null, null), 0.456);
    ndarray.ops.subseq(dataProcessedTensor.pick(0, 2, null, null), 0.406);
    ndarray.ops.divseq(dataProcessedTensor.pick(0, 0, null, null), 0.229);
    ndarray.ops.divseq(dataProcessedTensor.pick(0, 1, null, null), 0.224);
    ndarray.ops.divseq(dataProcessedTensor.pick(0, 2, null, null), 0.225);
    var tensor = new ort.Tensor('float32', new Float32Array(3 * width * height), [1, 3, width, height]);
    tensor.data.set(dataProcessedTensor.data);
    return tensor;
}