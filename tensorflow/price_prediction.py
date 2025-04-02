const tf = require("@tensorflow/tfjs-node");
const path = require("path");
const Product = require("../models/Product");

const MODEL_PATH = path.join(__dirname, "../models/pricePredictionModel");

/**
 * Predict the next price using a trained model.
 * @param {string} productId - The product ID from MongoDB.
 * @returns {Promise<number|null>}
 */
async function predictPrice(productId) {
    try {
        const product = await Product.findById(productId);
        if (!product || product.priceHistory.length < 2) {
            throw new Error("Not enough data for prediction.");
        }

        // Load the trained model
        const model = await tf.loadLayersModel(`file://${MODEL_PATH}`);

        // Predict the next price
        const lastPrice = tf.tensor2d([product.priceHistory[product.priceHistory.length - 1]], [1, 1]);
        const prediction = model.predict(lastPrice).dataSync()[0];

        console.log(`Predicted next price: ${prediction.toFixed(2)}`);
        return prediction.toFixed(2);
    } catch (error) {
        console.error("Error in price prediction:", error.message);
        return null;
    }
}

module.exports = { predictPrice };
