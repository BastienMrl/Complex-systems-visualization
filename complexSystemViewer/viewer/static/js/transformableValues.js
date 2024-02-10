const sizePerColor = 3;
const sizePerTranslation = 3;
export class TransformableValues {
    colors;
    translations;
    constructor(nbElements) {
        this.colors = new Float32Array(nbElements * sizePerColor).fill(0);
        this.translations = new Float32Array(nbElements * sizePerTranslation).fill(0);
    }
}
