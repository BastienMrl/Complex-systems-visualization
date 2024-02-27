export function hexToRgbA(hex) {
    let c;
    if (/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)) {
        c = hex.substring(1).split('');
        if (c.length == 3) {
            c = [c[0], c[0], c[1], c[1], c[2], c[2]];
        }
        c = '0x' + c.join('');
        return [((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255];
    }
    throw new Error('Bad Hex');
}
export function mapValue(inMin, inMax, outMin, outMax, value) {
    let ret = outMin + (value - inMin) * (outMax - outMin) / (inMax - inMin);
    console.log(ret);
    return ret;
}
