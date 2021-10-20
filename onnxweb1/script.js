const imageSize = 416;
async function main() {
    // jsut for drawing
    var image = document.getElementById("imageSource");
    var canvas = document.getElementById("myCanvas");
    // document.body.appendChild(canvas);
    canvas.width = 416;
    canvas.height = 416;
    var context = canvas.getContext("2d");
    context.drawImage(image, 0, 0, image.width, image.height, 0, 0, 416, 416);
    //actual
    try {
        const session = await ort.InferenceSession.create('./yolov4.onnx');
        const imageLoader = new ImageLoader(imageSize, imageSize);
        const imageData = await imageLoader.getImageData('./2.jpg');
        const width = imageSize;
        const height = imageSize;
        const preprocessedData = preprocess(imageData.data, width, height);

        const inputTensor = new ort.Tensor('float32', preprocessedData, [1, 3, width, height]);
        const feeds = { input: inputTensor };
        const results = await session.run(feeds);

        // read from results
        // const dataC = results.c.data;
        // document.write(`data of result tensor 'c': ${results}`);
        console.log(results);

    } catch (e) {
        document.write(`failed to inference ONNX model: ${e}.`);
    }
}

// main();

function preprocess(data, width, height) {
    const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, height, width]);
  
    // Normalize 0-255 to (-1)-1
    ndarray.ops.subseq(dataFromImage.pick(2, null, null), 103.939);
    ndarray.ops.subseq(dataFromImage.pick(1, null, null), 116.779);
    ndarray.ops.subseq(dataFromImage.pick(0, null, null), 123.68);
  
    // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
    ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));
    ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
    ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0));

    return dataProcessed.data;
}

class ImageLoader {
    constructor(imageWidth, imageHeight) {
      this.canvas = document.createElement('canvas');
      this.canvas.width = imageWidth;
      this.canvas.height = imageHeight;
      this.ctx = this.canvas.getContext('2d');
    }
    async getImageData(url) {
      await this.loadImageAsync(url);
      const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
      return imageData;
    }
    loadImageAsync(url) {
      return new Promise((resolve, reject) => {
        this.loadImageCb(url, () => {
          resolve();
        });
      });
    }
    loadImageCb(url, cb) {
      loadImage(
        url,
        img => {
          if (img.type === 'error') {
            throw `Could not load image: ${url}`;
          } else {
            // load image data onto input canvas
            this.ctx.drawImage(img, 0, 0)
            //console.log(`image was loaded`);
            window.setTimeout(() => { cb(); }, 0);
          }
        },
        {
          maxWidth: this.canvas.width,
          maxHeight: this.canvas.height,
          cover: true,
          crop: true,
          canvas: true,
          crossOrigin: 'Anonymous'
        }
      );
    }
  }