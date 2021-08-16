import * as maskedlm from "../maskedlm/pkg";  // Wasmをモジュールとして読み込み

const output = maskedlm.predict_masked_words("[MASK] is the capital city of Japan.");  // BERTでの推論実行
console.log(output)
