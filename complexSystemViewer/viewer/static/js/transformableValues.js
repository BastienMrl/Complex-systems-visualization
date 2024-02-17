const sizePerState = 1;
const sizePerTranslation = 3;
export class TransformableValues {
    _nbElement;
    states;
    translations;
    constructor(nbElements = 1) {
        this.reshape(nbElements);
    }
    static fromValues(states, translations) {
        let instance = new TransformableValues(states.length);
        instance.states = states;
        instance.translations = translations;
        return instance;
    }
    static fromInstance(values) {
        let instance = new TransformableValues(values.nbElements);
        instance.states = new Float32Array(values.states);
        instance.translations = new Float32Array(values.translations);
        return instance;
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
