// Define our labelmap
const labelMap = {
    1: { name: 'Hello', color: 'red' },
    2: { name: 'Yes', color: 'yellow' },
    3: { name: 'NO', color: 'lime' },
    4: { name: 'Thank_you', color: 'blue' },
    5: { name: 'I_Love_You', color: 'purple' },
}

const width_scale = 2
const length_scale = 1.5
 

// Define a drawing function
export const drawRect = (boxes, classes, scores, threshold, imgWidth, imgHeight, ctx) => {
    for (let i = 0; i <= boxes.length; i++) {
        if (boxes[i] && classes[i] && scores[i] > threshold) {
            // Extract variables
            const [y, x, height, width] = boxes[i]
            const text = classes[i]

            // Set styling
            ctx.strokeStyle = labelMap[text]['color']
            ctx.lineWidth = 10
            ctx.fillStyle = 'white'
            ctx.font = '30px Arial'
        
            if (labelMap[text]['name'] == "Hello"){

                const width_scale = 2.5
                const length_scale = 2

            } else if (labelMap[text]['name'] == "Yes") {
                const width_scale = 1.5
                const length_scale = 2.5

            } else if (labelMap[text]['name'] == "No"){
                
                const width_scale = 1.5
                const length_scale = 2.5

            } else if (labelMap[text]['name'] == "Thank_you"){
                
                const width_scale = 2
                const length_scale = 2.5
            } else if (labelMap[text]['name'] == "I_Love_You"){
                
                const width_scale = 1.2
                const length_scale = 1.5
            } else {
                
                const width_scale = 2
                const length_scale = 1.5
                 
            }

            // DRAW!!
            ctx.beginPath()
            ctx.fillText(labelMap[text]['name'] + ' - ' + Math.round(scores[i] * 100) / 100, x * imgWidth, y * imgHeight -10)
            
            ctx.rect(x * imgWidth, y * imgHeight, width * imgWidth /width_scale, height * imgHeight /length_scale);
            ctx.stroke()
        }
    }
}