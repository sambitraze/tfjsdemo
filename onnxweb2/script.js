const imageSize = 416;

const DEFAULT_FILTER_BOXES_THRESHOLD = 0.01;
const DEFAULT_IOU_THRESHOLD = 0.4;
const DEFAULT_CLASS_PROB_THRESHOLD = 0.3;
const INPUT_DIM = 416;

const classNames = ["Book", "Phone"];

const YOLO_ANCHORS = new ort.Tensor(
    'float32', Float32Array.from([
        0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434,
        7.88282, 3.52778, 9.77052, 9.16828,
    ]),
    [5, 2]);

async function main() {
    try {
        const session = await ort.InferenceSession.create('./yolov4.onnx');
        var image = document.getElementById("imageSource");
        var canvas = document.getElementById("myCanvas");
        // document.body.appendChild(canvas);
        canvas.width = 416;
        canvas.height = 416;
        var context = canvas.getContext("2d");
        const size = Math.min(image.width, image.height);
        const centerHeight = image.height / 2;
        const beginHeight = centerHeight - size / 2;
        const centerWidth = image.width / 2;
        const beginWidth = centerWidth - size / 2;
        context.drawImage(image, beginHeight, beginWidth, size, size, 0, 0, 416, 416);

        const preprocessedData = preProcess(context);
        const feeds = { input: preprocessedData };

        // feed inputs and run
        const results = await session.run(feeds);
        const output = results[session.outputNames[0]];
        await postprocess(output, context);

    } catch (e) {
        document.write(`failed to inference ONNX model: ${e}.`);
    }
}

// main();

function preProcess(ctx) {
    var imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    const { data, width, height } = imageData;
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [
        1,
        3,
        width,
        height,
    ]);

    ndarray.ops.assign(dataProcessedTensor.pick(0, 0, null, null), dataTensor.pick(null, null, 0));
    ndarray.ops.assign(dataProcessedTensor.pick(0, 1, null, null), dataTensor.pick(null, null, 1));
    ndarray.ops.assign(dataProcessedTensor.pick(0, 2, null, null), dataTensor.pick(null, null, 2));

    const tensor = new ort.Tensor("float32", new Float32Array(width * height * 3), [
        1,
        3,
        width,
        height,
    ]);
    tensor.data.set(dataProcessedTensor.data);
    return tensor;
}

async function postprocess(tensor, ctx) {
    try {
        const originalOutput = new ort.Tensor(
            "float32",
            new Float32Array(tensor.data),
            [1, 60, 13, 13]
        );
        const outputTensor = transpose(
            originalOutput,
            [0, 2, 3, 1]
        );

        // postprocessing
        const boxes = await yolopostprocess(outputTensor, 2);
        console.log(boxes.length)
        boxes.forEach((box) => {
            const { top, left, bottom, right, classProb, className } = box;
            console.log(className, classProb);
            drawRect(
                left,
                top,
                right - left,
                bottom - top,
                `${className} Confidence: ${Math.round(
                    classProb * 100
                )}%`, "red", ctx
            );
        });
    } catch (e) {
        console.log(e);
        alert(e);
    }
}

function drawRect(x, y, w, h, text = "", color = "red", ctx) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2
    ctx.fillStyle = 'white'
    ctx.font = '30px Arial'

    // DRAW!!
    ctx.beginPath()
    ctx.fillText(text, x, y);
    ctx.rect(x, y, w, h);
    ctx.stroke();
}

async function yolopostprocess(outputTensor, numClasses) {
    const [boxXy, boxWh, boxConfidence, boxClassProbs] = yolo_head(outputTensor, YOLO_ANCHORS, numClasses);
    const allBoxes = yolo_boxes_to_corners(boxXy, boxWh);
    const [outputBoxes, scores, classes] =
        await yolo_filter_boxes(allBoxes, boxConfidence, boxClassProbs, DEFAULT_FILTER_BOXES_THRESHOLD);
    // If all boxes have been filtered out    
    if (outputBoxes == null) {
        return [];
    }

    const width = scalar(INPUT_DIM);
    const height = scalar(INPUT_DIM);

    const imageDims = reshape(stack([height, width, height, width]), [
        1,
        4,
    ]);

    const boxes = mul(outputBoxes, imageDims);

    const [preKeepBoxesArr, scoresArr] = await Promise.all([
        boxes.data,
        scores.data,
    ]);

    const [keepIndx, boxesArr, keepScores] = non_max_suppression(
        preKeepBoxesArr, scoresArr,
        DEFAULT_IOU_THRESHOLD);

    const classesIndxArr = (await gather(classes, new ort.Tensor('int32', keepIndx)).data);

    const results = [];
    

    classesIndxArr.forEach((classIndx, i) => {
        const classProb = keepScores[i];
        if (classProb < DEFAULT_CLASS_PROB_THRESHOLD) {
            return;
        }

        const className = classNames[classIndx];
        let [top, left, bottom, right] = boxesArr[i];

        top = Math.max(0, top);
        left = Math.max(0, left);
        bottom = Math.min(416, bottom);
        right = Math.min(416, right);

        const resultObj = {
            className,
            classProb,
            bottom,
            top,
            left,
            right,
        };

        results.push(resultObj);
    });
    return results;
}

function gather(t, indices, axis = 0) {
    if (t.type === 'string') {
        throw new Error('Unspported type for this transformation');
    }
    if (indices.type !== 'int32' || (indices.dims && indices.dims.length > 1)) {
        throw new Error('Indices tensor not of specified format');
    }
    const dims = t.dims ? t.dims.slice() : [t.data.length];
    const newDims = dims;
    const indicesData = indices.data;
    newDims[axis] = indicesData.length;
    const dimsStrides = computeStrides(dims);
    const newDimsStrides = computeStrides(newDims);
    const Y = createTypedArray(t.type, ShapeUtilsize(newDims));
    const X = t.data;
    for (let i = 0; i < Y.length; ++i) {
        const newLogicalIndex = offsetToIndices(i, newDimsStrides);
        const oldLogicalIndex = newLogicalIndex.slice();
        oldLogicalIndex[axis] = indicesData[newLogicalIndex[axis]];
        const oldOffset = indicesToOffset(oldLogicalIndex, dimsStrides);
        Y[i] = X[oldOffset];
    }
    return new ort.Tensor(t.type, Y, newDims);
}
function indicesToOffset(indices, strides) {
    const rank = strides.length;
    if (rank === 0) {
        return 0;
    }
    let index = indices[indices.length - 1];
    for (let i = 0; i < indices.length - 1; ++i) {
        index += strides[i] * indices[i];
    }
    return index;
}
function offsetToIndices(offset, strides) {
    const rank = strides.length;
    if (rank === 0) {
        return [];
    }
    else if (rank === 1) {
        return [offset];
    }
    const indices = new Array(strides.length);
    for (let i = 0; i < indices.length - 1; ++i) {
        indices[i] = Math.floor(offset / strides[i]);
        offset -= indices[i] * strides[i];
    }
    indices[indices.length - 1] = offset;
    return indices;
}

function computeStrides(shape) {
    const rank = shape.length;
    if (rank < 2) {
        return [1];
    }

    const strides = new Array(rank);
    strides[rank - 1] = 1;
    strides[rank - 2] = shape[rank - 1];
    for (let i = rank - 3; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

function non_max_suppression(
    boxes, scores, iouThreshold) {
    // Zip together scores, box corners, and index
    const zipped = [];
    for (let i = 0; i < scores.length; i++) {
        zipped.push([
            scores[i],
            [boxes[4 * i], boxes[4 * i + 1], boxes[4 * i + 2], boxes[4 * i + 3]],
            i,
        ]);
    }
    // Sort by descending order of scores (first index of zipped array)
    const sortedBoxes = zipped.sort((a, b) => b[0] - a[0]);

    const selectedBoxes = [];

    // Greedily go through boxes in descending score order and only
    // return boxes that are below the IoU threshold.
    sortedBoxes.forEach((box) => {
        let add = true;
        for (let i = 0; i < selectedBoxes.length; i++) {
            // Compare IoU of zipped[1], since that is the box coordinates arr
            // TODO: I think there's a bug in this calculation
            const curIou = box_iou(box[1], selectedBoxes[i][1]);
            if (curIou > iouThreshold) {
                add = false;
                break;
            }
        }
        if (add) {
            selectedBoxes.push(box);
        }
    });

    // Return the kept indices and bounding boxes
    return [
        selectedBoxes.map((e) => e[2]),
        selectedBoxes.map((e) => e[1]),
        selectedBoxes.map((e) => e[0]),
    ];
}

function box_iou(a, b) {
    return box_intersection(a, b) / box_union(a, b);
}

function box_intersection(a, b) {
    const w = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
    const h = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
    if (w < 0 || h < 0) {
        return 0;
    }
    return w * h;
}
function box_union(a, b) {
    const i = box_intersection(a, b);
    return (a[3] - a[1]) * (a[2] - a[0]) + (b[3] - b[1]) * (b[2] - b[0]) - i;
}

async function yolo_filter_boxes(boxes, boxConfidence, boxClassProbs, threshold) {
    const boxScores = mul(boxConfidence, boxClassProbs);
    const boxClasses = argMax(boxScores, -1);
    const boxClassScores = max(boxScores, -1);
    // Many thanks to @jacobgil
    // Source: https://github.com/ModelDepot/tfjs-yolo-tiny/issues/6#issuecomment-387614801
    const predictionMask = as1D(greaterEqual(boxClassScores, scalar(threshold)));

    const N = predictionMask.size;
    // linspace start/stop is inclusive.
    const allIndices = cast(linspace(0, N - 1, N), 'int32');
    const negIndices = zeros([N], 'int32');
    const indices = where(predictionMask, allIndices, negIndices);

    return [
        gather(reshape(boxes, [N, 4]), indices),
        gather(as1D(boxClassScores), indices),
        gather(as1D(boxClasses), indices),
    ];
}

function validateDims(dims) {
    if (dims.length < 0 || dims.length > 6) {
        throw new TypeError(`Only rank 0 to 6 is supported for tensor shape.`);
    }

    if (dims.length === 0) {
        throw new RangeError('Scaler tensor is not implemented yet');
    }

    for (const n of dims) {
        if (!Number.isInteger(n)) {
            throw new TypeError(`Invalid shape: ${n} is not an integer`);
        }
        if (n <= 0 || n > 2147483647) {
            throw new TypeError(`Invalid shape: length ${n} is not allowed`);
        }
    }
}

function where(condition, t1, t2) {
    // validate shape and types of input tensors and condition tensor
    areEqual(t1.dims ? t1.dims : [t1.data.length], t2.dims ? t2.dims : [t2.data.length]);
    validateSameTypes([t1.type, t2.type]);
    if (condition.type !== 'bool') {
        throw new Error('Condition tensor must be bool type');
    }

    // create output
    const outputShape = t1.dims ? t1.dims : [t1.data.length];
    const output =
        new ort.Tensor(t1.type, createTypedArray(t1.type, ShapeUtilsize(outputShape)), outputShape);
    const outputData = output.data;

    // input data
    const conditionData = condition.data;
    const X = t1.data;
    const Y = t2.data;

    // condition is 1D rank
    if (!condition.dims || condition.dims.length === 1) {
        // the outermost dimension of the input tensors and condition tensor must be the same
        const conditionDims = condition.dims ? condition.dims : [condition.data.length];
        const t1Dims = t1.dims ? t1.dims : [t1.data.length];
        if (conditionDims[0] !== t1Dims[0]) {
            throw new Error('Outermost dimensions of input tensors and condition tensor must match');
        }

        let offset = 1;
        // Input tensors are not 1-D. Need to compute offset.
        if (t1.dims && t1.dims.length > 1) {
            for (let i = 1; i < t1.dims.length; ++i) {
                offset *= t1.dims[i];
            }
        }

        for (let i = 0; i < conditionData.length; ++i) {
            for (let j = 0; j < offset; ++j) {
                outputData[i * offset + j] = conditionData[i] > 0 ? X[i * offset + j] : Y[i * offset + j];
            }
        }
    } else {
        // The shapes of input tensors and condition tensor must be the same
        areEqual(condition.dims, t2.dims ? t2.dims : [t2.data.length]);

        for (let i = 0; i < conditionData.length; ++i) {
            outputData[i] = conditionData[i] > 0 ? X[i] : Y[i];
        }
    }
    return output;
}

function areEqual(shape1, shape2) {
    if (shape1.length !== shape2.length) {
        return false;
    }
    return shape1.every((v, i) => v === shape2[i]);
}

function zeros(dims, dtype = 'float32') {
    if (dtype !== 'float32' && dtype !== 'int32' && dtype !== 'bool') {
        throw new Error('Unsupported type for creating all zero Tensor');
    }
    validateDims(dims);
    return new ort.Tensor(dtype, createTypedArray(dtype, ShapeUtilsize(dims)), dims);
}

function linspace(start, stop, num) {
    if (num === 0) {
        throw new Error('Must request atleast one sample');
    }
    const increments = (stop - start) / (num - 1);
    const data = createTypedArray('float32', num);
    data[0] = start;
    for (let i = 1; i < data.length; i++) {
        data[i] = data[i - 1] + increments;
    }
    return new ort.Tensor('float32', data, [num]);
}

function scalar(value, dtype = 'float32') {
    if (dtype !== 'float32' && dtype !== 'int32') {
        throw new Error('Unsupported type for this transformation');
    }
    const data = createTypedArray(dtype, 1);
    data[0] = value;
    return new ort.Tensor(dtype, data, [1]);
}

function greaterEqual(t1, t2) {
    if ((t1.type !== 'float32' && t1.type !== 'int32' && t1.type !== 'bool') ||
        (t2.type !== 'float32' && t2.type !== 'int32' && t2.type !== 'bool')) {
        throw new Error('Unsupported type for transform');
    }
    if (t1.type !== t2.type) {
        throw new Error('Types are not homogeneous');
    }
    return binaryOp(t1, t2, (e1, e2) => (e1 >= e2 ? 1 : 0), 'bool');
}

function max(t, axis = 0, keepDims = false) {
    if (t.type !== 'float32' && t.type !== 'int32') {
        throw new Error('Unsupported type for transform');
    }
    const rank = t.dims ? t.dims.length : 1;
    axis = getActualAxisFromNegativeValue(axis, rank);
    const [reduceDims, resultDims] = splitDimsIntoTwo(t.dims ? t.dims : [t.data.length], axis);
    const X = t.data;
    const Y = createTypedArray(t.type, resultDims.length === 0 ? 1 : ShapeUtilsize(resultDims));
    const blockSize = reduceDims[0];
    for (let i = 0; i < Y.length; ++i) {
        const offset = blockSize * i;
        let max = X[offset];
        for (let j = 0; j < blockSize; ++j) {
            const value = X[offset + j];
            if (value > max) {
                max = value;
            }
        }
        Y[i] = max;
    }

    let adjustedResultDims = [];
    if (keepDims) {
        const origDims = t.dims ? t.dims : [t.data.length];
        for (let i = 0; i < origDims.length; ++i) {
            if (i === axis) {
                adjustedResultDims.push(1);
            } else {
                adjustedResultDims.push(origDims[i]);
            }
        }
    } else {
        adjustedResultDims = resultDims;
    }
    return new ort.Tensor(t.type, Y, adjustedResultDims.length === 0 ? [1] : adjustedResultDims);
}

function argMax(t, axis = 0) {
    if (t.type !== 'float32' && t.type !== 'int32') {
        throw new Error('Unsupported type for transform');
    }
    const rank = t.dims ? t.dims.length : 1;
    axis = getActualAxisFromNegativeValue(axis, rank);
    const [reduceDims, resultDims] = splitDimsIntoTwo(t.dims ? t.dims : [t.data.length], axis);
    const X = t.data;
    const Y = createTypedArray('int32', resultDims.length === 0 ? 1 : ShapeUtilsize(resultDims));
    const blockSize = reduceDims[0];
    for (let i = 0; i < Y.length; ++i) {
        const offset = blockSize * i;
        let max = X[offset];
        let index = 0;
        for (let j = 0; j < blockSize; ++j) {
            const value = X[offset + j];
            if (value > max) {
                max = value;
                index = j;
            }
        }
        Y[i] = index;
    }
    return new ort.Tensor('int32', Y, resultDims.length === 0 ? [1] : resultDims);
}

function splitDimsIntoTwo(dims, pick) {
    const picked = [];
    const remnants = [];

    for (let i = 0; i < dims.length; ++i) {
        if (i === pick) {
            picked.push(dims[i]);
        } else {
            remnants.push(dims[i]);
        }
    }

    return [picked, remnants];
}

function yolo_boxes_to_corners(boxXy, boxWh) {
    const two = new ort.Tensor('float32', [2.0]);
    const boxMins = sub(boxXy, div(boxWh, two));
    const boxMaxes = add(boxXy, div(boxWh, two));

    const dim0 = boxMins.dims[0];
    const dim1 = boxMins.dims[1];
    const dim2 = boxMins.dims[2];
    const size = [dim0, dim1, dim2, 1];

    return concat(
        [
            slice(boxMins, [0, 0, 0, 1], size),
            slice(boxMins, [0, 0, 0, 0], size),
            slice(boxMaxes, [0, 0, 0, 1], size),
            slice(boxMaxes, [0, 0, 0, 0], size),
        ],
        3);
}

function yolo_head(feats, anchors, numClasses) {
    const numAnchors = anchors.dims[0];

    const anchorsArray = reshape(anchors, [1, 1, numAnchors, 2]);

    const convDims = feats.dims.slice(1, 3);

    // For later use
    const convDims0 = convDims[0];
    const convDims1 = convDims[1];

    let convHeightIndex = range(0, convDims[0]);
    let convWidthIndex = range(0, convDims[1]);

    convHeightIndex = tile(convHeightIndex, [convDims[1]]);

    convWidthIndex = tile(expandDims(convWidthIndex, 0), [
        convDims[0],
        1,
    ]);
    convWidthIndex = as1D(transpose(convWidthIndex));

    let convIndex = transpose(stack([convHeightIndex, convWidthIndex]));
    convIndex = reshape(convIndex, [convDims[0], convDims[1], 1, 2]);
    convIndex = cast(convIndex, feats.type);

    feats = reshape(feats, [
        convDims[0],
        convDims[1],
        numAnchors,
        numClasses + 10,
    ]);
    const convDimsTensor = cast(reshape(new ort.Tensor('int32', convDims), [1, 1, 1, 2]), feats.type);

    let boxXy = sigmoid(slice(feats, [0, 0, 0, 0], [convDims0, convDims1, numAnchors, 2]));
    let boxWh = exp(slice(feats, [0, 0, 0, 2], [convDims0, convDims1, numAnchors, 2]));
    const boxConfidence = sigmoid(slice(feats, [0, 0, 0, 4], [convDims0, convDims1, numAnchors, 1]));
    const boxClassProbs = softmax(slice(feats, [0, 0, 0, 5], [convDims0, convDims1, numAnchors, numClasses]));

    boxXy = div(add(boxXy, convIndex), convDimsTensor);
    boxWh = div(mul(boxWh, anchorsArray), convDimsTensor);

    return [boxXy, boxWh, boxConfidence, boxClassProbs];
}
function exp(t) {
    if (t.type !== 'float32' && t.type !== 'int32') {
        throw new Error('Unsupported type for transform');
    }
    return unaryOpsexp(t);
}
function unaryOpsexp(input) {
    if (input.type === 'string') {
        throw new Error('Unsupported type for transform');
    }
    const X = input.data;
    const Y = createTypedArray(input.type, X.length);
    for (let i = 0; i < X.length; i++) {
        Y[i] = Math.exp(X[i]);
    }
    return new ort.Tensor(input.type, Y, input.dims ? input.dims : [input.data.length]);
}
function mul(t1, t2) {
    if ((t1.type !== 'float32' && t1.type !== 'int32') || (t2.type !== 'float32' && t2.type !== 'int32')) {
        throw new Error('Unsupported type for transform');
    }
    if (t1.type !== t2.type) {
        throw new Error('Types are not homogeneous');
    }
    return binaryOp(t1, t2, (e1, e2) => (e1 * e2), t1.type);
}


function sub(t1, t2) {
    if ((t1.type !== 'float32' && t1.type !== 'int32') || (t2.type !== 'float32' && t2.type !== 'int32')) {
        throw new Error('Unsupported type for transform');
    }
    if (t1.type !== t2.type) {
        throw new Error('Types are not homogeneous');
    }
    return binaryOp(t1, t2, (e1, e2) => (e1 - e2), t1.type);
}

function div(t1, t2) {
    if ((t1.type !== 'float32' && t1.type !== 'int32') || (t2.type !== 'float32' && t2.type !== 'int32')) {
        throw new Error('Unsupported type for transform');
    }
    if (t1.type !== t2.type) {
        throw new Error('Types are not homogeneous');
    }
    // TODO: Handle division by zero if any
    return binaryOp(t1, t2, (e1, e2) => (e1 / e2), t1.type);
}

function add(t1, t2) {
    if ((t1.type !== 'float32' && t1.type !== 'int32') || (t2.type !== 'float32' && t2.type !== 'int32')) {
        throw new Error('Unsupported type for transform');
    }
    if (t1.type !== t2.type) {
        throw new Error('Types are not homogeneous');
    }
    return binaryOp(t1, t2, (e1, e2) => (e1 + e2), t1.type);
}
function binaryOp(
    x, y, opLambda, resultType) {
    const result = calc(
        ndarray(x.data, x.dims ? x.dims.slice(0) : [x.data.length]),
        ndarray(y.data, y.dims ? y.dims.slice(0) : [y.data.length]), opLambda);
    if (!result) {
        throw new Error('not broadcastable');
    }
    const rType = resultType ? resultType : x.type;
    const output =
        new ort.Tensor(rType, rType === 'bool' ? Uint8Array.from(result.data) : result.data, result.shape);
    return output;
}

function calc(a, b, op) {
    const shape = calcShape(a.shape, b.shape);
    if (shape) {
        const size = ShapeUtilsize(shape);
        const c = ndarray(
            new (
                a.data.constructor)(size),
            shape);

        const indices = new Array(shape.length);
        for (let i = 0; i < size; i++) {
            // traversal indices
            let rest = i;
            for (let j = shape.length - 1; j >= 0; j--) {
                indices[j] = rest % shape[j];
                rest = Math.floor(rest / shape[j]);
            }

            // map index
            const indicesA = index(indices, a.shape);
            const indicesB = index(indices, b.shape);

            // assign value
            c.set(...indices.concat(op(a.get(...indicesA), b.get(...indicesB))));
        }

        return c;
    }

    return undefined;
}

function index(indices, shapeOrigin, isMatMul = false) {
    // we assume the parameter indices is valid. ie. it should have the same
    // length as the broadcasted shape, and for each dimension the index should
    // not be out of range.
    const dimOffset = indices.length - shapeOrigin.length;
    const indicesOrigin = indices.slice(dimOffset);
    const dimLen = isMatMul ? indicesOrigin.length - 2 : indicesOrigin.length;
    for (let i = 0; i < dimLen; i++) {
        indicesOrigin[i] = indices[dimOffset + i] % shapeOrigin[i];
    }
    return indicesOrigin;
}

function calcShape(adims, bdims, isMatMul = false) {
    const arank = adims.length;
    const brank = bdims.length;
    const crank = Math.max(adims.length, bdims.length);
    const cdims = new Array(crank);

    // calculate the last 2 dimension if it is MatMul
    if (isMatMul) {
        if (arank < 2 || brank < 2) {
            return undefined;
        }
        const cShapeMatMul =
            calcMatMulShape([adims[arank - 2], adims[arank - 1]], [bdims[brank - 2], bdims[brank - 1]]);
        if (cShapeMatMul === undefined) {
            return undefined;
        }
        [cdims[crank - 2], cdims[crank - 1]] = cShapeMatMul;
    }

    for (let i = isMatMul ? 3 : 1; i <= crank; i++) {
        const aLen = arank - i < 0 ? 1 : adims[arank - i];
        const bLen = brank - i < 0 ? 1 : bdims[brank - i];

        if (aLen !== bLen && aLen > 1 && bLen > 1) {
            return undefined;
        }
        cdims[crank - i] = Math.max(aLen, bLen);
    }

    return cdims;
}
function calcMatMulShape(a, b) {
    return (a[1] !== b[0]) ? undefined : [a[0], b[1]];
}
function softmax(x, axis) {
    const inputDimensions = x.dims ? x.dims : [x.data.length];
    const inputRank = inputDimensions.length;

    const axisCorrected = getActualAxisFromNegativeValue(axis, inputRank);
    const N = sizeToDimension(inputDimensions, axisCorrected);
    const D = sizeFromDimension(inputDimensions, axisCorrected);

    const X = x.data;

    const Y = createTypedArray(x.type, x.data.length);

    for (let i = 0; i < N; i++) {
        // find row offset
        const offset = i * D;

        // find max of each logical row
        let max = Number.MIN_VALUE;
        for (let j = 0; j < D; j++) {
            if (X[offset + j] > max) {
                max = X[offset + j];
            }
        }

        // find normalization scale per row
        let scale = 0;
        for (let j = 0; j < D; j++) {
            const value = X[offset + j] - max;
            Y[offset + j] = Math.exp(value);
            scale += Math.exp(value);
        }

        // perform the softmax normalization
        for (let j = 0; j < D; j++) {
            if (scale === 0) {
                Y[offset + j] = 0;
            } else {
                Y[offset + j] /= scale;
            }
        }
    }

    return new ort.Tensor(x.type, Y, inputDimensions);
}
function sizeToDimension(dims, axis) {
    if (axis > dims.length) {
        throw new Error(`invalid dimension of ${axis} for sizeToDimension as Tensor has ${dims.length} dimensions.`);
    }

    return getSizeFromDimensionRange(dims, 0, axis);
}
function getSizeFromDimensionRange(dims, start, end) {
    let size = 1;
    for (let i = start; i < end; i++) {
        // safety check as this method is called by multiple other methods requiring size.
        // size cannot be 0 or negative.
        if (dims[i] <= 0) {
            throw new Error(
                // tslint:disable-next-line:max-line-length
                `cannot get valid size from specified dimension range. Most likely the range contains 0 or negative values in them.`);
        }
        size *= dims[i];
    }
    return size;
}
function slice(t, begin, size) {
    if (t.type === 'string') {
        throw new Error('Unspported type for this transformation');
    }
    const newDimsStride = computeStrides(size);
    const oldDimsStride = computeStrides(t.dims ? t.dims : [t.data.length]);
    const X = t.data;
    const Y = createTypedArray(t.type, ShapeUtilsize(size));
    for (let i = 0; i < Y.length; ++i) {
        const newLogicalIndex = offsetToIndices(i, newDimsStride);
        const oldLogicalIndex = newLogicalIndex.map((idx, j) => idx + begin[j]);
        const oldOffset = indicesToOffset(oldLogicalIndex, oldDimsStride);
        Y[i] = X[oldOffset];
    }
    return new ort.Tensor(t.type, Y, size);
}

function sigmoid(input) {
    const X = input.data;
    const Y = createTypedArray(input.type, X.length);
    for (let i = 0; i < X.length; i++) {
        Y[i] = (1 / (1 + Math.exp(-X[i])));
    }
    return new ort.Tensor(input.type, Y, input.dims ? input.dims : [input.data.length]);
}

function cast(t, dtype) {
    // TODO: If the requested type and the given type are the same, return same tensor ?
    // Need to investigate if it breaks some basic assumptions
    switch (dtype) {
        case 'int32':
            return new ort.Tensor('int32', Int32Array.from(t.data), t.dims ? t.dims : [t.data.length]);
        case 'float32':
            return new ort.Tensor('float32', Float32Array.from(t.data), t.dims ? t.dims : [t.data.length]);
        case 'bool':
            return new ort.Tensor('bool', Uint8Array.from(t.data), t.dims ? t.dims : [t.data.length]);
        default:
            throw new Error('Unsupported type for casting');
    }
}
function stack(tensors, axis = 0) {
    if (tensors.length < 2) {
        throw new Error('Must have atleast 2 tensors to stack');
    }

    const types = [];
    const shapes = [];
    tensors.forEach(t => {
        types.push(t.type);
        shapes.push(t.dims ? t.dims : [t.data.length]);
    });
    validateSameTypes(types);
    validateEqualDims(shapes);
    const rank = tensors[0].dims ? tensors[0].dims.length : 1;
    axis = getActualAxisFromNegativeValue(axis, rank);
    const expanded = tensors.map(t => expandDims(t, axis));
    return concat(expanded, axis, false);
}
function concat(tensors, axis = 0, typeCheckRequired = true) {
    if (tensors.length < 2) {
        throw new Error('Must have atleast 2 tensors to concatenate');
    }

    if (typeCheckRequired) {
        const types = [];
        tensors.forEach(t => {
            types.push(t.type);
        });
        validateSameTypes(types);
    }

    return concatImpl(tensors, axis);
}
function concatImpl(x, axis) {
    const input0 = x[0];
    const inputShape = input0.dims ? input0.dims : [input0.data.length];

    if (axis >= inputShape.length || axis < (-1 * inputShape.length)) {
        throw new Error(`axis specified for concat doesn't match input dimensionality`);
    }

    if (axis < 0) {
        axis = inputShape.length + axis;
    }

    // ensure all of the non-concatenated axes match each other
    // along the way, calculate the shape of the output tensor
    let concatAxisSize = inputShape[axis];
    const outputShape = new Array(inputShape.length);

    for (let i = 1; i < x.length; i++) {
        const dataN = x[i];
        const dataNShape = dataN.dims ? dataN.dims : [dataN.data.length];

        for (let axisIndex = 0; axisIndex < inputShape.length; axisIndex++) {
            // add to the placeholder for computing output shape
            if (axisIndex === axis) {
                concatAxisSize += dataNShape[axisIndex];
            }

            // ensure all non-cancatenated axes match each other
            if (inputShape[axisIndex] !== dataNShape[axisIndex]) {
                throw new Error(`non concat dimensions must match`);
            }

            // fill the 'outputShape' array
            outputShape[axisIndex] = dataNShape[axisIndex];
        }
    }

    // complete the 'outputShape' array
    outputShape[axis] = concatAxisSize;

    // main logic
    // tslint:disable-next-line:max-line-length
    const output =
        new ort.Tensor(input0.type, createTypedArray(x[0].type, ShapeUtilsize(outputShape)), outputShape);
    const Y = output.data;

    // the axisPitch is the number of elements to add to move
    // to the next split axis in the output
    let axisPitch = 1;
    for (let i = outputShape.length - 1; i >= axis; i--) {
        axisPitch *= outputShape[i];
    }

    let outputBase = 0;
    for (let inputIndex = 0; inputIndex < x.length; inputIndex++) {
        const dataN = x[inputIndex];
        const dataNDims = dataN.dims ? dataN.dims : [dataN.data.length];

        // the inputAxisPitch is the number of elements to add
        // to move to the next split axis in the input
        let inputAxisPitch = 1;
        for (let i = dataNDims.length - 1; i >= axis; i--) {
            inputAxisPitch *= dataNDims[i];
        }

        const inputData = dataN.data;
        const inputSize = ShapeUtilsize(dataNDims);

        // copy the data across.
        // for every 'inputAxisPitch' values copied, we move over by
        // the 'axisPitch'

        let outputOffset = outputBase;

        for (let i = 0, j = 0; i < inputSize; i++) {
            Y[outputOffset + i] = inputData[i];
            if (++j === inputAxisPitch) {
                // subtract inputAxisPitch because output is being indexed by 'i'
                outputOffset += (axisPitch - inputAxisPitch);
                j = 0;
            }
        }
        outputBase += inputAxisPitch;
    }

    return output;
}
function validateEqualDims(dimsArray) {
    if (dimsArray.length < 2) {
        throw new Error('must contain atleast 2 shapes to compare equality');
    }
    const baseDims = dimsArray[0];
    const baseRank = baseDims.length;
    for (let i = 1; i < dimsArray.length; ++i) {
        const dims = dimsArray[i];
        if (dims.length !== baseRank) {
            throw new Error('rank is not the same for given inpu shapes');
        }
        for (let j = 0; j < baseRank; ++j) {
            if (baseDims[j] !== dims[j]) {
                throw new Error('input shapes are not the same');
            }
        }
    }
}
function validateSameTypes(typesArray) {
    if (typesArray.length < 2) {
        throw new Error('must contain atleast 2 types to compare equality');
    }
    const baseType = typesArray[0];
    for (let i = 0; i < typesArray.length; ++i) {
        if (typesArray[i] !== baseType) {
            throw new Error('input types are ');
        }
    }
}

function as1D(t) {
    return reshape(t, [t.data.length]);
}

function expandDims(t, axis = 0) {
    axis = getActualAxisFromNegativeValue(axis, t.dims ? t.dims.length : 1);
    const dims = t.dims ? t.dims : [t.data.length];
    const changedShapeLength = dims.length + 1;
    const changedShape = new Array(changedShapeLength);
    let iter = 0;
    for (let i = 0; i < changedShapeLength; ++i) {
        if (i === axis) {
            changedShape[i] = 1;
        } else {
            changedShape[i] = dims[iter++];
        }
    }
    return new ort.Tensor(t.type, t.data, changedShape);
}

function getActualAxisFromNegativeValue(axis, tensorRank) {
    if (axis < -tensorRank && axis >= tensorRank - 1) {
        throw new Error('unsupported axis for this operation.');
    }
    return axis < 0 ? axis + tensorRank : axis;
}

function tile(t, reps) {
    if (t.type === 'string') {
        throw new Error('Unspported type for this transformation');
    }
    const dims = t.dims ? t.dims : [t.data.length];
    const rank = dims.length;
    const newDims = new Array(rank);
    if (rank !== reps.length) {
        throw new Error('Repetitions must be of the same rank as input dims');
    }
    for (let i = 0; i < rank; i++) {
        newDims[i] = dims[i] * reps[i];
    }
    const dimsStrides = computeStrides(dims);
    const newDimsStrides = computeStrides(newDims);
    const Y = createTypedArray(t.type, ShapeUtilsize(newDims));
    const X = t.data;
    for (let i = 0; i < Y.length; ++i) {
        const newLogicalIndex = offsetToIndices(i, newDimsStrides);
        const oldLogicalIndex = new Array(rank);
        for (let j = 0; j < rank; ++j) {
            oldLogicalIndex[j] = newLogicalIndex[j] % t.dims[j];
        }
        const oldOffset = indicesToOffset(oldLogicalIndex, dimsStrides);
        Y[i] = X[oldOffset];
    }
    return new ort.Tensor(t.type, Y, newDims);
}

function range(start, stop, step = 1, dtype = 'float32') {
    if (step === 0) {
        throw new Error('Step size of 0 is not acceptable');
    }
    // adjust default values
    if (stop < step && step === 1) {
        step = -1;
    }
    // the following conditions cannot generate any data
    if (start === step || (start < stop && step < 0) || (stop < start && step > 0)) {
        return new ort.Tensor(dtype, createTypedArray(dtype, 1), [1]);
    }
    const size = Math.abs(Math.ceil((stop - start) / step));
    const data = createTypedArray(dtype, size);
    data[0] = start;
    for (let i = 1; i < data.length; i++) {
        data[i] = data[i - 1] + step;
    }
    return new ort.Tensor(dtype, data, [size]);
}

function reshape(x, shape) {
    const reshapedDims = calculateReshapedDims(x.dims, shape);
    const output = new ort.Tensor(x.type, createTypedArray(x.type, x.data.length), reshapedDims);
    const X = x.data;
    const Y = output.data;
    for (let i = 0; i < x.data.length; ++i) {
        Y[i] = X[i];
    }
    return output;
}

function calculateReshapedDims(originalDims, shapeHints) {
    const nDims = shapeHints.length;
    const reshapedDims = new Array(nDims);
    let unknownDimension = -1;
    let size = 1;

    for (let i = 0; i < nDims; i++) {
        if (shapeHints[i] < -1) {
            throw new Error('a dimension cannot be less than -1');
        }
        if (shapeHints[i] === -1) {
            if (unknownDimension !== -1) {
                throw new Error('at most one dimension can be -1');
            }
            unknownDimension = i;
        } else {
            if (shapeHints[i] === 0) {
                if (i >= originalDims.length) {
                    throw new Error('the dimension with value zero exceeds the dimension size of the input tensor');
                }
                reshapedDims[i] = originalDims[i];
            } else {
                reshapedDims[i] = shapeHints[i];
            }
            size *= reshapedDims[i];
        }
    } if (unknownDimension !== -1) {
        const originalTensorFlattenedSize = ShapeUtil.size(originalDims);
        if (originalTensorFlattenedSize % size !== 0) {
            throw new Error(`the input tensor cannot be reshaped to the requested shape. Input shape: [${originalDims}] Output shape: [${shapeHints}]`);
        }
        reshapedDims[unknownDimension] = originalTensorFlattenedSize / size;
    }
    return reshapedDims;
}

function transpose(x, perm) {
    const inputDims = x.dims ? x.dims : [x.data.length];
    const rank = inputDims.length;
    // determine permutation to use
    // if no permutation was specified in the attributes,
    // the default is [rank-1, ..., 0]
    let finalPerm = new Array(rank);
    if (perm && perm.length === rank) {
        finalPerm = perm;
    }
    else {
        for (let i = 0; i < rank; i++) {
            finalPerm[i] = rank - i - 1;
        }
    }
    const outputDims = new Array(rank);
    const stride = new Array(rank);
    // determine shape of output, as well as stride to be used
    // stride[i] indicates the stride for the input-tensor dimension
    // corresponding to the i-th dimension of the output
    for (let i = 0; i < rank; i++) {
        const inpDim = finalPerm[i];
        outputDims[i] = inputDims[inpDim];
        if (inpDim + 1 < rank) {
            stride[i] = sizeFromDimension(inputDims, inpDim + 1);
        }
        else {
            stride[i] = 1;
        }
    }
    const output = new ort.Tensor(x.type, createTypedArray(x.type, ShapeUtilsize(outputDims)), outputDims);
    const X = x.data;
    const Y = output.data;
    // partition the permutation into a prefix and the largest suffix such that
    // every axis i in the suffix is mapped to i.
    let numAxesInPrefix = 0; // number of axes in prefix
    let suffixBlocksize = 1; // product of dimensions in the suffix
    let prefixBlocksize = 1; // product of dimensions in the prefix
    let isSuffix = true;
    for (let i = rank - 1; i >= 0; --i) {
        const inpAxis = finalPerm[i];
        if (isSuffix && (inpAxis === i)) {
            suffixBlocksize *= inputDims[inpAxis];
        }
        else {
            isSuffix = false;
            prefixBlocksize *= inputDims[inpAxis];
            ++numAxesInPrefix;
        }
    }
    if (prefixBlocksize === 1) {
        doTransposeSingleBlock(suffixBlocksize, Y, X);
    }
    else if (suffixBlocksize === 1) {
        doTransposeEltWise(numAxesInPrefix, outputDims, prefixBlocksize, stride, Y, X);
    }
    else {
        doTranspose(numAxesInPrefix, outputDims, prefixBlocksize, suffixBlocksize, stride, Y, X);
    }
    return output;
}

function doTransposeSingleBlock(numElementsInBlock, target, source) {
    arrayCopyHelper(target, source, 0, 0, numElementsInBlock);
}

function doTransposeEltWise(
    numAxes, targetDims, numBlocks, stride, target,
    source) {
    const targetIndex = new Array(numAxes).fill(0);

    let startTargetIndex = 0;

    for (let i = 0; i < numBlocks; ++i) {
        const sourceOffset = computeOffset(targetIndex, stride, numAxes);
        target[startTargetIndex++] = source[sourceOffset];
        incrementIndex(targetIndex, targetDims, numAxes);
    }
}

function doTranspose(
    numAxes, targetDims, numBlocks, numElementsInBlock, stride,
    target, source) {
    const targetIndex = new Array(numAxes).fill(0);

    const startSourceIndex = 0;
    let startTargetIndex = 0;

    for (let i = 0; i < numBlocks; ++i) {
        const sizeOffset = computeOffset(targetIndex, stride, numAxes);
        arrayCopyHelper(target, source, startTargetIndex, startSourceIndex + sizeOffset, numElementsInBlock);

        incrementIndex(targetIndex, targetDims, numAxes);
        startTargetIndex += numElementsInBlock;
    }
}

function incrementIndex(index, dims, axisToIncrementOn) {
    if (axisToIncrementOn === undefined) {
        axisToIncrementOn = dims.length;
    }

    for (let k = axisToIncrementOn - 1; k >= 0; --k) {
        index[k]++;
        if (index[k] < dims[k]) {
            break;
        }
        index[k] = 0;
    }
}
function computeOffset(index, stride, axis) {
    if (axis === undefined) {
        axis = index.length;
    }
    let offset = 0;
    for (let i = 0; i < axis; ++i) {
        offset += (index[i] * stride[i]);
    }
    return offset;
}

function arrayCopyHelper(
    target, source, targetIndex, sourceIndex, blockSize) {
    if (sourceIndex < 0 || sourceIndex >= source.length) {
        throw new Error(`sourceIndex out of bounds`);
    }
    if (targetIndex < 0 || targetIndex >= target.length) {
        throw new Error(`targetIndex out of bounds`);
    }
    if (sourceIndex + blockSize > source.length) {
        throw new Error(`source indices to be copied are outside bounds`);
    }
    if (targetIndex + blockSize > target.length) {
        throw new Error(`target array is too small to hold result`);
    }

    for (let offset = 0; offset < blockSize; offset++) {
        target[targetIndex + offset] = source[sourceIndex + offset];
    }
}

function sizeFromDimension(dims, axis) {
    if (axis > dims.length) {
        throw new Error(`invalid dimension of ${axis} for sizeFromDimension as Tensor has ${dims.length} dimensions.`);
    }
    return getSizeFromDimensionRange(dims, axis, dims.length);
}

function getSizeFromDimensionRange(dims, start, end) {
    let size = 1;
    for (let i = start; i < end; i++) {
        // safety check as this method is called by multiple other methods requiring size.
        // size cannot be 0 or negative.
        if (dims[i] <= 0) {
            throw new Error(
                // tslint:disable-next-line:max-line-length
                `cannot get valid size from specified dimension range. Most likely the range contains 0 or negative values in them.`);
        }
        size *= dims[i];
    }
    return size;
}

function ShapeUtilsize(dims) {
    return getSizeFromDimensionRange(dims, 0, dims.length);
}

function createTypedArray(type, size) {
    switch (type) {
        case 'bool':
            return new Uint8Array(size);
        case 'int32':
            return new Int32Array(size);
        case 'float32':
            return new Float32Array(size);
        default:
            throw new Error('Unsupported type');
    }
}

main();