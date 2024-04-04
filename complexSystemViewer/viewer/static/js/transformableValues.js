const sizePerState = 1;
const sizePerTranslation = 3;
const idxDomain = 0;
const idxX = 1;
const idxY = 2;
const idxFirstState = 3;
const idxNbElements = 0;
const idxNbChannels = 1;
const idxMinX = 2;
const idxMaxX = 3;
const idxMinY = 4;
const idxMaxY = 5;
const idxDomainStatesFirst = 6;
export class TransformableValues {
    states;
    translations;
    domain;
    constructor(domain = new Float32Array([0, 0, 0, 0, 0, 0])) {
        this.domain = domain;
        this.reshape();
    }
    static fromValuesAsArray(array) {
        let instance = new TransformableValues(array[0]);
        instance.translations = array[1];
        for (let i = 0; i < instance.nbChannels; i++) {
            instance.states[i] = array[i + 2];
        }
        return instance;
    }
    static fromInstance(values) {
        let instance = new TransformableValues(new Float32Array(values.domain));
        instance.states = new Array(values.nbChannels);
        values.states.forEach((e, i) => instance.states[i] = new Float32Array(e));
        instance.translations = new Float32Array(values.translations);
        return instance;
    }
    setWithBackendValues(values) {
        values[idxDomain].forEach((e, i) => {
            this.domain[i] = e;
        });
        for (let i = 0; i < this.nbChannels; i++) {
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
        let ret = new Array(this.nbChannels + 2);
        for (let i = 0; i < this.nbElements; ++i) {
            posX[i] = this.translations[i * 3];
            posY[i] = this.translations[i * 3 + 1];
        }
        ret[0] = posX;
        ret[1] = posY;
        for (let i = 0; i < this.nbChannels; i++) {
            ret[i + 2] = this.states[i];
        }
        return ret;
    }
    toArray() {
        let ret = new Array(this.nbChannels + 2);
        for (let i = 0; i < this.nbChannels; i++) {
            ret[i + 2] = this.states[i];
        }
        ret[0] = this.domain;
        ret[1] = this.translations;
        return ret;
    }
    toArrayBuffers() {
        let ret = new Array(this.nbChannels + 2);
        for (let i = 0; i < this.nbChannels; i++) {
            ret[i + 2] = this.states[i].buffer;
        }
        ret[0] = this.domain.buffer;
        ret[1] = this.translations.buffer;
        return ret;
    }
    get nbElements() {
        return this.domain[idxNbElements];
    }
    get nbChannels() {
        return this.domain[idxNbChannels];
    }
    getBoundsX() {
        return [this.domain[idxMinX], this.domain[idxMaxX]];
    }
    getBoundsY() {
        return [this.domain[idxMinY], this.domain[idxMaxY]];
    }
    getBoundsStates(idx) {
        if (idx < 0 || idx >= this.nbChannels)
            return;
        return [this.domain[idxDomainStatesFirst + idx * 2], this.domain[idxDomainStatesFirst + idx * 2 + 1]];
    }
    reshape() {
        this.states = new Array(this.nbChannels);
        for (let i = 0; i < this.nbChannels; i++) {
            this.states[i] = new Float32Array(this.nbElements * sizePerState).fill(0);
        }
        this.translations = new Float32Array(this.nbElements * sizePerTranslation).fill(0.);
    }
    reinitTranslation() {
        this.translations = new Float32Array(this.nbElements * sizePerTranslation).fill(0.);
    }
}
