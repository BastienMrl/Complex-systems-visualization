const sizePerState = 1;
const sizePerTranslation = 3;
const idxDomain = 0;
const idxX = 1;
const idxY = 2;
const idxFirstState = 3;
export class TransformableValues {
    _nbElement;
    _nbChannels;
    states;
    translations;
    constructor(nbElements = 1, nbChannels = 1) {
        this.reshape(nbElements, nbChannels);
    }
    static fromValuesAsArray(array) {
        let instance = new TransformableValues(array[0].length / sizePerTranslation, array.length - 1);
        instance.translations = array[0];
        for (let i = 0; i < instance._nbChannels; i++) {
            instance.states[i] = array[i + 1];
        }
        return instance;
    }
    static fromInstance(values) {
        let instance = new TransformableValues(values.nbElements);
        instance.states = new Array(values._nbChannels);
        values.states.forEach((e, i) => instance.states[i] = new Float32Array(e));
        instance.translations = new Float32Array(values.translations);
        return instance;
    }
    setWithBackendValues(values) {
        for (let i = 0; i < this._nbChannels; i++) {
            this.states[i] = new Float32Array(values[i + idxFirstState]);
        }
        values[idxX].forEach((e, i) => {
            this.translations[i * 3] = e;
        });
        values[idxY].forEach((e, i) => {
            this.translations[i * 3 + 1] = e;
        });
    }
    getBackendValues() {
        let posX = new Float32Array(this.nbElements);
        let posY = new Float32Array(this.nbElements);
        let ret = new Array(this._nbChannels + 2);
        for (let i = 0; i < this.nbElements; ++i) {
            posX[i] = this.translations[i * 3];
            posY[i] = this.translations[i * 3 + 1];
        }
        ret[0] = posX;
        ret[1] = posY;
        for (let i = 0; i < this._nbChannels; i++) {
            ret[i + 2] = this.states[i];
        }
        return ret;
    }
    toArray() {
        let ret = new Array(this._nbChannels + 1);
        for (let i = 0; i < this._nbChannels; i++) {
            ret[i + 1] = this.states[i];
        }
        ret[0] = this.translations;
        return ret;
    }
    toArrayBuffers() {
        let ret = new Array(this._nbChannels + 1);
        for (let i = 0; i < this._nbChannels; i++) {
            ret[i + 1] = this.states[i].buffer;
        }
        ret[0] = this.translations.buffer;
        return ret;
    }
    get nbElements() {
        return this._nbElement;
    }
    get nbChannels() {
        return this._nbChannels;
    }
    reshape(nbElements, nbChannels) {
        if (nbElements != undefined)
            this._nbElement = nbElements;
        if (nbChannels != undefined)
            this._nbChannels = nbChannels;
        this.states = new Array(nbElements);
        for (let i = 0; i < nbChannels; i++) {
            this.states[i] = new Float32Array(nbElements * sizePerState).fill(0);
        }
        this.translations = new Float32Array(nbElements * sizePerTranslation).fill(0.);
    }
    reinitTranslation() {
        this.translations = new Float32Array(this.nbElements * sizePerTranslation).fill(0.);
    }
}
