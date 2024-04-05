const sizePerState = 1;
const sizePerTranslation = 3;
const idxDomain = 0;
const idxX = 1;
const idxY = 2;
const idxFirstState = 3;
const idxId = 0;
const idxNbElements = 1;
const idxNbChannels = 2;
const idxMinX = 3;
const idxMaxX = 4;
const idxMinY = 5;
const idxMaxY = 6;
const idxDomainStatesFirst = 7;
export class TransformableValues {
    states;
    positionX;
    positionY;
    domain;
    constructor(domain = new Float32Array([0, 0, 0, 0, 0, 0, 0])) {
        this.domain = domain;
        this.reshape();
    }
    static fromValuesAsArray(array) {
        let instance = new TransformableValues(array[0]);
        instance.positionX = array[1];
        instance.positionY = array[2];
        for (let i = 0; i < instance.nbChannels; i++) {
            instance.states[i] = array[i + 3];
        }
        return instance;
    }
    static fromInstance(values) {
        let instance = new TransformableValues(new Float32Array(values.domain));
        instance.states = new Array(values.nbChannels);
        values.states.forEach((e, i) => instance.states[i] = new Float32Array(e));
        instance.positionX = new Float32Array(values.positionX);
        instance.positionY = new Float32Array(values.positionY);
        return instance;
    }
    setWithBackendValues(values) {
        values[idxDomain].forEach((e, i) => {
            this.domain[i] = e;
        });
        for (let i = 0; i < this.nbChannels; i++) {
            this.states[i] = new Float32Array(values[i + idxFirstState]);
        }
        this.positionX = new Float32Array(values[idxX]);
        this.positionY = new Float32Array(values[idxY]);
    }
    getBackendValues() {
        let posX = new Float32Array(this.positionX);
        let posY = new Float32Array(this.positionY);
        let ret = new Array(this.nbChannels + 2);
        ret[0] = posX;
        ret[1] = posY;
        for (let i = 0; i < this.nbChannels; i++) {
            ret[i + 2] = this.states[i];
        }
        return ret;
    }
    toArray() {
        let ret = new Array(this.nbChannels + 3);
        for (let i = 0; i < this.nbChannels; i++) {
            ret[i + 3] = this.states[i];
        }
        ret[0] = this.domain;
        ret[1] = this.positionX;
        ret[2] = this.positionY;
        return ret;
    }
    toArrayBuffers() {
        let ret = new Array(this.nbChannels + 3);
        for (let i = 0; i < this.nbChannels; i++) {
            ret[i + 3] = this.states[i].buffer;
        }
        ret[0] = this.domain.buffer;
        ret[1] = this.positionX.buffer;
        ret[2] = this.positionY.buffer;
        return ret;
    }
    get nbElements() {
        return this.domain[idxNbElements];
    }
    get nbChannels() {
        return this.domain[idxNbChannels];
    }
    get id() {
        return this.domain[idxId];
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
        this.positionX = new Float32Array(this.nbElements).fill(0.);
        this.positionY = new Float32Array(this.nbElements).fill(0.);
    }
}
