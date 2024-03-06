const sizePerState = 1;
const sizePerTranslation = 3;
export class TransformableValues {
    _nbElement;
    states;
    translations;
    constructor(nbElements = 1) {
        this.reshape(nbElements);
    }
    static fromArray(array) {
        let instance = new TransformableValues(array[0].length / sizePerTranslation);
        instance.translations = array[0];
        instance.states = array[1];
        return instance;
    }
    static fromInstance(values) {
        let instance = new TransformableValues(values.nbElements);
        instance.states = new Float32Array(values.states);
        instance.translations = new Float32Array(values.translations);
        return instance;
    }
    setWithBackendValues(values) {
        this.states = new Float32Array(values[2]);
        values[0].forEach((e, i) => {
            this.translations[i * 3] = e;
        });
        values[1].forEach((e, i) => {
            this.translations[i * 3 + 1] = e;
        });
    }
    getBackendValues() {
        let states = new Float32Array(this.states);
        let posX = new Float32Array(states.length);
        let posY = new Float32Array(states.length);
        for (let i = 0; i < states.length; ++i) {
            posX[i] = this.translations[i * 3];
            posY[i] = this.translations[i * 3 + 1];
        }
        return [posX, posY, states];
    }
    toArray() {
        return [this.translations, this.states];
    }
    toArrayBuffers() {
        return [this.translations.buffer, this.states.buffer];
    }
    get nbElements() {
        return this._nbElement;
    }
    reshape(nbElements) {
        this._nbElement = nbElements;
        this.states = new Float32Array(nbElements * sizePerState).fill(0.);
        this.translations = new Float32Array(nbElements * sizePerTranslation).fill(0.);
    }
    reinitTranslation() {
        this.translations = new Float32Array(this.nbElements * sizePerTranslation).fill(0.);
    }
}
