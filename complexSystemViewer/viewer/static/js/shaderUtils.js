async function getShaderFromFile(src, type, gl) {
    const response = await fetch(src, { cache: "no-cache" });
    const text = await response.text();
    return getShaderFromString(text, type, gl);
}
function getShaderFromString(src, type, gl) {
    const shader = gl.createShader(type);
    if (shader == null) {
        throw "Could not create Shader from source files";
    }
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        throw `Could not compile WebGL program. \n\n${info}`;
    }
    return shader;
}
async function initShaders(gl, srcVertex, srcFragment) {
    const vertexShader = await getShaderFromFile(srcVertex, gl.VERTEX_SHADER, gl);
    const fragmentShader = await getShaderFromFile(srcFragment, gl.FRAGMENT_SHADER, gl);
    let shaderProgram = gl.createProgram();
    if (shaderProgram == null) {
        throw "Cour not create Shader Program";
    }
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.transformFeedbackVaryings(shaderProgram, ['feedback_translation'], gl.SEPARATE_ATTRIBS);
    gl.linkProgram(shaderProgram);
    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        alert("Could not initialise shaders");
    }
    return shaderProgram;
}
export class ProgramWithTransformer {
    _context;
    _program;
    _templateVertexShader;
    _currentTransformers;
    _fragmentShader;
    _vertexShader;
    static _transformersKey = "//${TRANSFORMERS}";
    constructor(context) {
        this._context = context;
    }
    get program() {
        return this._program;
    }
    async generateProgram(srcVertex, srcFragment) {
        this._fragmentShader = await fetch(srcFragment, { cache: "no-cache" }).then(response => response.text());
        this._templateVertexShader = await fetch(srcVertex, { cache: "no-cache" }).then(response => response.text());
        if (this._currentTransformers != undefined)
            this._vertexShader = this._templateVertexShader.replace(ProgramWithTransformer._transformersKey, this._currentTransformers);
        else
            this._vertexShader = this._templateVertexShader;
        this.reloadProgram();
    }
    updateProgramTransformers(transformers) {
        this._currentTransformers = transformers;
        if (this._templateVertexShader == undefined)
            return;
        this._vertexShader = this._templateVertexShader.replace(ProgramWithTransformer._transformersKey, this._currentTransformers);
        this.reloadProgram();
    }
    reloadProgram() {
        let vertexShader = getShaderFromString(this._vertexShader, this._context.VERTEX_SHADER, this._context);
        let fragmentShader = getShaderFromString(this._fragmentShader, this._context.FRAGMENT_SHADER, this._context);
        let shaderProgram = this._context.createProgram();
        if (shaderProgram == null) {
            throw "Cour not create Shader Program";
        }
        this._context.attachShader(shaderProgram, vertexShader);
        this._context.attachShader(shaderProgram, fragmentShader);
        this._context.linkProgram(shaderProgram);
        if (!this._context.getProgramParameter(shaderProgram, this._context.LINK_STATUS)) {
            alert("Could not initialise shaders");
        }
        this._program = shaderProgram;
    }
}
export function getAnimableValueUniformName(value) {
    switch (value) {
        case AnimableValue.COLOR:
            return ShaderUniforms.TIME_COLOR;
        case AnimableValue.POSITION:
            return ShaderUniforms.TIME_TRANSLATION;
        case AnimableValue.ROTATION:
            return ShaderUniforms.TIME_ROTATION;
        case AnimableValue.SCALING:
            return ShaderUniforms.TIME_SCALING;
    }
}
var AnimableValue;
(function (AnimableValue) {
    AnimableValue[AnimableValue["COLOR"] = 0] = "COLOR";
    AnimableValue[AnimableValue["POSITION"] = 1] = "POSITION";
    AnimableValue[AnimableValue["ROTATION"] = 2] = "ROTATION";
    AnimableValue[AnimableValue["SCALING"] = 3] = "SCALING";
})(AnimableValue || (AnimableValue = {}));
var ShaderUniforms;
(function (ShaderUniforms) {
    ShaderUniforms["TIME_COLOR"] = "u_time_color";
    ShaderUniforms["TIME_TRANSLATION"] = "u_time_translation";
    ShaderUniforms["TIME_ROTATION"] = "u_time_rotation";
    ShaderUniforms["TIME_SCALING"] = "u_time_scaling";
})(ShaderUniforms || (ShaderUniforms = {}));
var ShaderMeshInputs;
(function (ShaderMeshInputs) {
    ShaderMeshInputs["TRANSLATION_T0"] = "a_translation_t0";
    ShaderMeshInputs["TRANLSATION_T1"] = "a_translation_t1";
    ShaderMeshInputs["STATE_0_T0"] = "a_state_0_t0";
    ShaderMeshInputs["STATE_0_T1"] = "a_state_0_t1";
    ShaderMeshInputs["STATE_1_T0"] = "a_state_1_t0";
    ShaderMeshInputs["STATE_1_T1"] = "a_state_1_t1";
    ShaderMeshInputs["STATE_2_T0"] = "a_state_2_t0";
    ShaderMeshInputs["STATE_2_T1"] = "a_state_2_t1";
    ShaderMeshInputs["STATE_3_T0"] = "a_state_3_t0";
    ShaderMeshInputs["STATE_3_T1"] = "a_state_3_t1";
})(ShaderMeshInputs || (ShaderMeshInputs = {}));
var ShaderVariable;
(function (ShaderVariable) {
    ShaderVariable["COLOR"] = "color";
    ShaderVariable["TRANSLATION"] = "translation";
    ShaderVariable["SCALING"] = "scaling";
    ShaderVariable["ROTATION"] = "rotation";
})(ShaderVariable || (ShaderVariable = {}));
var ShaderFunction;
(function (ShaderFunction) {
    ShaderFunction["FACTOR"] = "factor_transformer";
    ShaderFunction["INTERPOLATION"] = "interpolation_transformer";
    ShaderFunction["INPUT_FROM_TIME"] = "get_input_value_from_time";
    ShaderFunction["NORMALIZE_POSITION"] = "normalize_position";
})(ShaderFunction || (ShaderFunction = {}));
var ShaderLocation;
(function (ShaderLocation) {
    ShaderLocation[ShaderLocation["POS"] = 0] = "POS";
    ShaderLocation[ShaderLocation["NORMAL"] = 1] = "NORMAL";
    ShaderLocation[ShaderLocation["UV"] = 2] = "UV";
    ShaderLocation[ShaderLocation["ID"] = 3] = "ID";
    ShaderLocation[ShaderLocation["SELECTED"] = 4] = "SELECTED";
    ShaderLocation[ShaderLocation["TRANSLATION_T0"] = 5] = "TRANSLATION_T0";
    ShaderLocation[ShaderLocation["TRANLSATION_T1"] = 6] = "TRANLSATION_T1";
    ShaderLocation[ShaderLocation["STATE_0_T0"] = 7] = "STATE_0_T0";
    ShaderLocation[ShaderLocation["STATE_0_T1"] = 8] = "STATE_0_T1";
    ShaderLocation[ShaderLocation["STATE_1_T0"] = 9] = "STATE_1_T0";
    ShaderLocation[ShaderLocation["STATE_1_T1"] = 10] = "STATE_1_T1";
    ShaderLocation[ShaderLocation["STATE_2_T0"] = 11] = "STATE_2_T0";
    ShaderLocation[ShaderLocation["STATE_2_T1"] = 12] = "STATE_2_T1";
    ShaderLocation[ShaderLocation["STATE_3_T0"] = 13] = "STATE_3_T0";
    ShaderLocation[ShaderLocation["STATE_3_T1"] = 14] = "STATE_3_T1";
})(ShaderLocation || (ShaderLocation = {}));
export { initShaders, ShaderVariable, ShaderFunction, ShaderMeshInputs, ShaderUniforms, ShaderLocation, AnimableValue };
