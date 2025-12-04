// Written by Dor Verbin, October 2021
// This is based on: http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// With additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/

function playVids3(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge3");
    var vid = document.getElementById(videoId);
    console.log(videoId + "Merge3");

    var pos_array = new Array();
    pos_array[0] = 0.333;
    pos_array[1] = 0.666;
    var flag = false;
    var line_flag = new Array();
    line_flag[0] = false;
    line_flag[1] = false;
    var vidWidth = vid.videoWidth/3;
    var vidHeight = vid.videoHeight;

    var mergeContext = videoMerge.getContext("2d");

    
    if (vid.readyState > 3) {
        vid.play();

        function trackLocation(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.pageX - bcr.x) / bcr.width);
        }
        function trackLocationTouch(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            tmp = ((e.pageX - bcr.x) / bcr.width)
            if(flag == true){
                // console.log(line_flag[0], line_flag[1]);
                if(Math.abs(tmp - pos_array[0])<= 0.1 && Math.abs(tmp - pos_array[1])<= 0.1){
                    if(line_flag[0] == false && line_flag[1] == false){
                        if(Math.abs(tmp - pos_array[0]) < Math.abs(tmp - pos_array[1])){line_flag[0] = true; line_flag[1] = false;}
                        else{line_flag[1] = true; line_flag[0] = false;}
                    }
                    if(line_flag[0] == true){
                        pos_array[0] = tmp;
                        if(tmp>pos_array[1]) pos_array[1] = tmp;
                    }else{
                        pos_array[1] = tmp;
                        if(tmp<pos_array[0]) pos_array[0] = tmp;
                    }
                }else{
                    if(Math.abs(tmp - pos_array[0])<= 0.1) {
                        if(line_flag[0] == false && line_flag[1] == false){
                            pos_array[0] = tmp; line_flag[0] = true; line_flag[1] = false;
                        }else if(line_flag[0] == false){
                            if(tmp<pos_array[0]) pos_array[0] = tmp;
                        }else{
                            pos_array[0] = tmp;
                        }
                    }
                    if(Math.abs(tmp - pos_array[1])<= 0.1) {
                        if(line_flag[0] == false && line_flag[1] == false){
                            pos_array[1] = tmp; line_flag[1] = true; line_flag[0] = false;
                        }else if(line_flag[1] == false){
                            if(tmp>pos_array[1]) pos_array[1] = tmp;
                        }else{
                            pos_array[1] = tmp;
                        }
                    }
                }
            }
        }

        function setflag(e){
            // console.log(e);
            if(e.type == "mousedown") flag = true;
            else{
                flag = false;
                line_flag[0] = false;
                line_flag[1] = false;
            }
        }

        videoMerge.addEventListener("mousemove",  trackLocationTouch, false); 
        videoMerge.addEventListener("mousedown", setflag, false);
        videoMerge.addEventListener("mouseup", setflag, false);
        videoMerge.addEventListener("mouseleave", setflag, false);


        function drawLoop() {
            mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);
            var colStart = (vidWidth * pos_array[0]).clamp(0.0, vidWidth);
            var colWidth = Math.abs((vidWidth * pos_array[1]) - (vidWidth * pos_array[0])).clamp(0.0, vidWidth);
            var colStart2 = (vidWidth * pos_array[1]).clamp(0.0, vidWidth);
            var colWidth2 = Math.abs(vidWidth - (vidWidth * pos_array[1])).clamp(0.0, vidWidth);
            mergeContext.drawImage(vid, colStart+vidWidth, 0, colWidth, vidHeight, colStart, 0, colWidth, vidHeight);
            mergeContext.drawImage(vid, colStart2+vidWidth*2, 0, colWidth2, vidHeight, colStart2, 0, colWidth2, vidHeight);
            requestAnimationFrame(drawLoop);

            
            var arrowLength = 0.06 * vidHeight;
            var arrowheadWidth = 0.025 * vidHeight;
            var arrowheadLength = 0.04 * vidHeight;
            var arrowPosY = vidHeight / 10;
            var arrowWidth = 0.007 * vidHeight;
            var currX_array = new Array();
            currX_array[0] = vidWidth * pos_array[0];
            currX_array[1] = vidWidth * pos_array[1];

            for (i=0; i<currX_array.length; ++i){
                // Draw circle
                mergeContext.arc(currX_array[i], arrowPosY, arrowLength*0.7, 0, Math.PI * 2, false);
                mergeContext.fillStyle = "#FFD79340";
                mergeContext.fill()
                //mergeContext.strokeStyle = "#444444";
                //mergeContext.stroke()
                
                // Draw border
                mergeContext.beginPath();
                mergeContext.moveTo(vidWidth*pos_array[i], 0);
                mergeContext.lineTo(vidWidth*pos_array[i], vidHeight);
                mergeContext.closePath()
                mergeContext.strokeStyle = "#AAAAAA";
                mergeContext.lineWidth = 3;            
                mergeContext.stroke();
                
                // Draw arrow
                mergeContext.beginPath();
                mergeContext.moveTo(currX_array[i], arrowPosY - arrowWidth/3);
                
                // Move right until meeting arrow head
                mergeContext.lineTo(currX_array[i] + arrowLength/3 - arrowheadLength/3, arrowPosY - arrowWidth/3);
                
                // Draw right arrow head
                mergeContext.lineTo(currX_array[i] + arrowLength/3 - arrowheadLength/3, arrowPosY - arrowheadWidth/3);
                mergeContext.lineTo(currX_array[i] + arrowLength/3, arrowPosY);
                mergeContext.lineTo(currX_array[i] + arrowLength/3 - arrowheadLength/3, arrowPosY + arrowheadWidth/3);
                mergeContext.lineTo(currX_array[i] + arrowLength/3 - arrowheadLength/3, arrowPosY + arrowWidth/3);
                
                // Go back to the left until meeting left arrow head
                mergeContext.lineTo(currX_array[i] - arrowLength/3 + arrowheadLength/3, arrowPosY + arrowWidth/3);
                
                // Draw left arrow head
                mergeContext.lineTo(currX_array[i] - arrowLength/3 + arrowheadLength/3, arrowPosY + arrowheadWidth/3);
                mergeContext.lineTo(currX_array[i] - arrowLength/3, arrowPosY);
                mergeContext.lineTo(currX_array[i] - arrowLength/3 + arrowheadLength/3, arrowPosY  - arrowheadWidth/3);
                mergeContext.lineTo(currX_array[i] - arrowLength/3 + arrowheadLength/3, arrowPosY);
                
                mergeContext.lineTo(currX_array[i] - arrowLength/3 + arrowheadLength/3, arrowPosY - arrowWidth/3);
                mergeContext.lineTo(currX_array[i], arrowPosY - arrowWidth/3);
                
                mergeContext.closePath();
                
                mergeContext.fillStyle = "#AAAAAA";
                mergeContext.fill();
            }
        }
        requestAnimationFrame(drawLoop);
    } 
}


function playVids4(videoId) {
    // 获取 canvas 元素和视频元素（注意 canvas 的 id 为 videoId+"Merge4"）
    var videoMerge = document.getElementById(videoId + "Merge4");
    var vid = document.getElementById(videoId);
    console.log(videoId + "Merge4");

    // 初始化三个分界值（相对于每个小画面的比例）
    var pos_array = [0.25, 0.5, 0.75];
    var flag = false;
    // 对应每个分界值的拖拽状态标记
    var line_flag = [false, false, false];

    // 将视频宽度按 4 分割得到每个段落的宽度
    var vidWidth = vid.videoWidth / 4;
    var vidHeight = vid.videoHeight;

    var mergeContext = videoMerge.getContext("2d");

    if (vid.readyState > 3) {
        vid.play();

        // 处理鼠标拖拽更新分界值（仅处理离 canvas 的 pageX 坐标）
        function trackLocationTouch(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var tmp = (e.pageX - bcr.x) / bcr.width;
            if (flag) {
                // 在所有分界点中寻找与当前鼠标位置最近且在 0.1 阈值内的那个
                var closestIndex = -1;
                var minDiff = 1.0;
                for (var i = 0; i < pos_array.length; i++) {
                    var diff = Math.abs(tmp - pos_array[i]);
                    if (diff <= 0.1 && diff < minDiff) {
                        minDiff = diff;
                        closestIndex = i;
                    }
                }
                if (closestIndex >= 0) {
                    // 为防止拖拽时分界线彼此交叉，对拖拽结果进行夹取：
                    if (closestIndex === 0) {
                        tmp = Math.min(tmp, pos_array[1]);
                    } else if (closestIndex === pos_array.length - 1) {
                        tmp = Math.max(tmp, pos_array[closestIndex - 1]);
                    } else {
                        tmp = Math.max(pos_array[closestIndex - 1], Math.min(tmp, pos_array[closestIndex + 1]));
                    }
                    pos_array[closestIndex] = tmp;
                    line_flag[closestIndex] = true;
                }
            }
        }

        function setflag(e) {
            if (e.type === "mousedown") {
                flag = true;
            } else {
                flag = false;
                for (var i = 0; i < line_flag.length; i++) {
                    line_flag[i] = false;
                }
            }
        }

        videoMerge.addEventListener("mousemove", trackLocationTouch, false);
        videoMerge.addEventListener("mousedown", setflag, false);
        videoMerge.addEventListener("mouseup", setflag, false);
        videoMerge.addEventListener("mouseleave", setflag, false);

        function drawLoop() {
            // 清空 canvas，如果需要可增加 clearRect：
            // mergeContext.clearRect(0, 0, videoMerge.width, videoMerge.height);

            // 绘制第一个片段（原视频的第 1 段，源区域 [0, vidWidth]）
            mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);

            // 计算剩余三个片段中需要剪裁合成的区域：
            // —— 第二段：从视频第 2 个区域中取 [vidWidth*pos_array[0], vidWidth*pos_array[1]]
            var colStart1 = (vidWidth * pos_array[0]).clamp(0, vidWidth);
            var colWidth1 = Math.abs((vidWidth * pos_array[1]) - (vidWidth * pos_array[0])).clamp(0, vidWidth);
            // —— 第三段：从视频第 3 个区域中取 [vidWidth*pos_array[1], vidWidth*pos_array[2]]
            var colStart2 = (vidWidth * pos_array[1]).clamp(0, vidWidth);
            var colWidth2 = Math.abs((vidWidth * pos_array[2]) - (vidWidth * pos_array[1])).clamp(0, vidWidth);
            // —— 第四段：从视频第 4 个区域中取 [vidWidth*pos_array[2], vidWidth]
            var colStart3 = (vidWidth * pos_array[2]).clamp(0, vidWidth);
            var colWidth3 = Math.abs(vidWidth - (vidWidth * pos_array[2])).clamp(0, vidWidth);

            // 分别从视频的 2～4 段中剪裁并绘制到 canvas 上（目标坐标和尺寸与源区域对应）
            mergeContext.drawImage(vid, colStart1 + vidWidth, 0, colWidth1, vidHeight, colStart1, 0, colWidth1, vidHeight);
            mergeContext.drawImage(vid, colStart2 + vidWidth * 2, 0, colWidth2, vidHeight, colStart2, 0, colWidth2, vidHeight);
            mergeContext.drawImage(vid, colStart3 + vidWidth * 3, 0, colWidth3, vidHeight, colStart3, 0, colWidth3, vidHeight);

            // 绘制拖拽用的标记（圆圈、竖线和箭头）
            var arrowLength = 0.06 * vidHeight;
            var arrowheadWidth = 0.025 * vidHeight;
            var arrowheadLength = 0.04 * vidHeight;
            var arrowPosY = vidHeight / 10;
            var arrowWidth = 0.007 * vidHeight;
            var currX_array = [];
            for (var i = 0; i < pos_array.length; i++) {
                currX_array[i] = vidWidth * pos_array[i];
            }

            for (var i = 0; i < currX_array.length; i++) {
                // 绘制圆形标记
                mergeContext.beginPath();
                mergeContext.arc(currX_array[i], arrowPosY, arrowLength * 0.7, 0, Math.PI * 2, false);
                mergeContext.fillStyle = "#FFD79340";
                mergeContext.fill();

                // 绘制垂直分隔线
                mergeContext.beginPath();
                mergeContext.moveTo(vidWidth * pos_array[i], 0);
                mergeContext.lineTo(vidWidth * pos_array[i], vidHeight);
                mergeContext.closePath();
                mergeContext.strokeStyle = "#AAAAAA";
                mergeContext.lineWidth = 3;
                mergeContext.stroke();

                // 绘制箭头图形
                mergeContext.beginPath();
                mergeContext.moveTo(currX_array[i], arrowPosY - arrowWidth / 3);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3 - arrowheadLength / 3, arrowPosY - arrowWidth / 3);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3 - arrowheadLength / 3, arrowPosY - arrowheadWidth / 3);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3, arrowPosY);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3 - arrowheadLength / 3, arrowPosY + arrowheadWidth / 3);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3 - arrowheadLength / 3, arrowPosY + arrowWidth / 3);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY + arrowWidth / 3);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY + arrowheadWidth / 3);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3, arrowPosY);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY - arrowheadWidth / 3);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY - arrowWidth / 3);
                mergeContext.lineTo(currX_array[i], arrowPosY - arrowWidth / 3);
                mergeContext.closePath();
                mergeContext.fillStyle = "#AAAAAA";
                mergeContext.fill();
            }
            requestAnimationFrame(drawLoop);
        }
        requestAnimationFrame(drawLoop);
    }
}





Number.prototype.clamp = function(min, max) {
  return Math.min(Math.max(this, min), max);
};
    



function resizeAndPlay3(element)
{
  var cv = document.getElementById(element.id + "Merge3");
  cv.width = element.videoWidth/3;
  cv.height = element.videoHeight;
  element.play();
  element.style.height = "0px";  // Hide video without stopping it
    
  playVids3(element.id);
}

function resizeAndPlay4(element)
{
  var cv = document.getElementById(element.id + "Merge4");
  cv.width = element.videoWidth/4;
  cv.height = element.videoHeight;
  element.play();
  element.style.height = "0px";  // Hide video without stopping it
    
  playVids4(element.id);
}

function playVids5(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge5");
    var vid = document.getElementById(videoId);
    console.log(videoId + "Merge5");

    var pos_array = [0.2, 0.4, 0.6, 0.8];
    var flag = false;
    var line_flag = [false, false, false, false];

    var vidWidth = vid.videoWidth / 5;
    var vidHeight = vid.videoHeight;

    var mergeContext = videoMerge.getContext("2d");

    if (vid.readyState > 3) {
        vid.play();

        function trackLocationTouch(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var tmp = (e.pageX - bcr.x) / bcr.width;
            if (flag) {
                var closestIndex = -1;
                var minDiff = 1.0;
                for (var i = 0; i < pos_array.length; i++) {
                    var diff = Math.abs(tmp - pos_array[i]);
                    if (diff <= 0.1 && diff < minDiff) {
                        minDiff = diff;
                        closestIndex = i;
                    }
                }
                if (closestIndex >= 0) {
                    if (closestIndex === 0) {
                        tmp = Math.min(tmp, pos_array[1]);
                    } else if (closestIndex === pos_array.length - 1) {
                        tmp = Math.max(tmp, pos_array[closestIndex - 1]);
                    } else {
                        tmp = Math.max(pos_array[closestIndex - 1], Math.min(tmp, pos_array[closestIndex + 1]));
                    }
                    pos_array[closestIndex] = tmp;
                    line_flag[closestIndex] = true;
                }
            }
        }

        function setflag(e) {
            if (e.type === "mousedown") {
                flag = true;
            } else {
                flag = false;
                for (var i = 0; i < line_flag.length; i++) {
                    line_flag[i] = false;
                }
            }
        }

        videoMerge.addEventListener("mousemove", trackLocationTouch, false);
        videoMerge.addEventListener("mousedown", setflag, false);
        videoMerge.addEventListener("mouseup", setflag, false);
        videoMerge.addEventListener("mouseleave", setflag, false);

        function drawLoop() {
            mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);

            var colStart1 = (vidWidth * pos_array[0]).clamp(0, vidWidth);
            var colWidth1 = Math.abs((vidWidth * pos_array[1]) - (vidWidth * pos_array[0])).clamp(0, vidWidth);
            var colStart2 = (vidWidth * pos_array[1]).clamp(0, vidWidth);
            var colWidth2 = Math.abs((vidWidth * pos_array[2]) - (vidWidth * pos_array[1])).clamp(0, vidWidth);
            var colStart3 = (vidWidth * pos_array[2]).clamp(0, vidWidth);
            var colWidth3 = Math.abs((vidWidth * pos_array[3]) - (vidWidth * pos_array[2])).clamp(0, vidWidth);
            var colStart4 = (vidWidth * pos_array[3]).clamp(0, vidWidth);
            var colWidth4 = Math.abs(vidWidth - (vidWidth * pos_array[3])).clamp(0, vidWidth);

            mergeContext.drawImage(vid, colStart1 + vidWidth, 0, colWidth1, vidHeight, colStart1, 0, colWidth1, vidHeight);
            mergeContext.drawImage(vid, colStart2 + vidWidth * 2, 0, colWidth2, vidHeight, colStart2, 0, colWidth2, vidHeight);
            mergeContext.drawImage(vid, colStart3 + vidWidth * 3, 0, colWidth3, vidHeight, colStart3, 0, colWidth3, vidHeight);
            mergeContext.drawImage(vid, colStart4 + vidWidth * 4, 0, colWidth4, vidHeight, colStart4, 0, colWidth4, vidHeight);

            var arrowLength = 0.06 * vidHeight;
            var arrowheadWidth = 0.025 * vidHeight;
            var arrowheadLength = 0.04 * vidHeight;
            var arrowPosY = vidHeight / 10;
            var arrowWidth = 0.007 * vidHeight;
            var currX_array = pos_array.map(pos => vidWidth * pos);

            for (var i = 0; i < currX_array.length; i++) {
                mergeContext.beginPath();
                mergeContext.arc(currX_array[i], arrowPosY, arrowLength * 0.7, 0, Math.PI * 2, false);
                mergeContext.fillStyle = "#FFD79340";
                mergeContext.fill();

                mergeContext.beginPath();
                mergeContext.moveTo(vidWidth * pos_array[i], 0);
                mergeContext.lineTo(vidWidth * pos_array[i], vidHeight);
                mergeContext.closePath();
                mergeContext.strokeStyle = "#AAAAAA";
                mergeContext.lineWidth = 3;
                mergeContext.stroke();

                mergeContext.beginPath();
                mergeContext.moveTo(currX_array[i], arrowPosY - arrowWidth / 3);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3 - arrowheadLength / 3, arrowPosY - arrowWidth / 3);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3 - arrowheadLength / 3, arrowPosY - arrowheadWidth / 3);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3, arrowPosY);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3 - arrowheadLength / 3, arrowPosY + arrowheadWidth / 3);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3 - arrowheadLength / 3, arrowPosY + arrowWidth / 3);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY + arrowWidth / 3);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY + arrowheadWidth / 3);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3, arrowPosY);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY - arrowheadWidth / 3);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY - arrowWidth / 3);
                mergeContext.lineTo(currX_array[i], arrowPosY - arrowWidth / 3);
                mergeContext.closePath();
                mergeContext.fillStyle = "#AAAAAA";
                mergeContext.fill();
            }
            requestAnimationFrame(drawLoop);
        }
        requestAnimationFrame(drawLoop);
    }
}

function resizeAndPlay5(element)
{
  var cv = document.getElementById(element.id + "Merge5");
  cv.width = element.videoWidth/5;
  cv.height = element.videoHeight;
  element.play();
  element.style.height = "0px";  // Hide video without stopping it
    
  playVids5(element.id);
}

function playVids6(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge6");
    var vid = document.getElementById(videoId);
    console.log(videoId + "Merge6");

    var pos_array = [0.166, 0.333, 0.5, 0.666, 0.833];
    var flag = false;
    var line_flag = [false, false, false, false, false];

    var vidWidth = vid.videoWidth / 6;
    var vidHeight = vid.videoHeight;

    var mergeContext = videoMerge.getContext("2d");

    if (vid.readyState > 3) {
        vid.play();

        function trackLocationTouch(e) {
            var bcr = videoMerge.getBoundingClientRect();
            var tmp = (e.pageX - bcr.x) / bcr.width;
            if (flag) {
                var closestIndex = -1;
                var minDiff = 1.0;
                for (var i = 0; i < pos_array.length; i++) {
                    var diff = Math.abs(tmp - pos_array[i]);
                    if (diff <= 0.1 && diff < minDiff) {
                        minDiff = diff;
                        closestIndex = i;
                    }
                }
                if (closestIndex >= 0) {
                    if (closestIndex === 0) {
                        tmp = Math.min(tmp, pos_array[1]);
                    } else if (closestIndex === pos_array.length - 1) {
                        tmp = Math.max(tmp, pos_array[closestIndex - 1]);
                    } else {
                        tmp = Math.max(pos_array[closestIndex - 1], Math.min(tmp, pos_array[closestIndex + 1]));
                    }
                    pos_array[closestIndex] = tmp;
                    line_flag[closestIndex] = true;
                }
            }
        }

        function setflag(e) {
            if (e.type === "mousedown") {
                flag = true;
            } else {
                flag = false;
                for (var i = 0; i < line_flag.length; i++) {
                    line_flag[i] = false;
                }
            }
        }

        videoMerge.addEventListener("mousemove", trackLocationTouch, false);
        videoMerge.addEventListener("mousedown", setflag, false);
        videoMerge.addEventListener("mouseup", setflag, false);
        videoMerge.addEventListener("mouseleave", setflag, false);

        function drawLoop() {
            mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);

            var colStart1 = (vidWidth * pos_array[0]).clamp(0, vidWidth);
            var colWidth1 = Math.abs((vidWidth * pos_array[1]) - (vidWidth * pos_array[0])).clamp(0, vidWidth);
            var colStart2 = (vidWidth * pos_array[1]).clamp(0, vidWidth);
            var colWidth2 = Math.abs((vidWidth * pos_array[2]) - (vidWidth * pos_array[1])).clamp(0, vidWidth);
            var colStart3 = (vidWidth * pos_array[2]).clamp(0, vidWidth);
            var colWidth3 = Math.abs((vidWidth * pos_array[3]) - (vidWidth * pos_array[2])).clamp(0, vidWidth);
            var colStart4 = (vidWidth * pos_array[3]).clamp(0, vidWidth);
            var colWidth4 = Math.abs((vidWidth * pos_array[4]) - (vidWidth * pos_array[3])).clamp(0, vidWidth);
            var colStart5 = (vidWidth * pos_array[4]).clamp(0, vidWidth);
            var colWidth5 = Math.abs(vidWidth - (vidWidth * pos_array[4])).clamp(0, vidWidth);

            mergeContext.drawImage(vid, colStart1 + vidWidth, 0, colWidth1, vidHeight, colStart1, 0, colWidth1, vidHeight);
            mergeContext.drawImage(vid, colStart2 + vidWidth * 2, 0, colWidth2, vidHeight, colStart2, 0, colWidth2, vidHeight);
            mergeContext.drawImage(vid, colStart3 + vidWidth * 3, 0, colWidth3, vidHeight, colStart3, 0, colWidth3, vidHeight);
            mergeContext.drawImage(vid, colStart4 + vidWidth * 4, 0, colWidth4, vidHeight, colStart4, 0, colWidth4, vidHeight);
            mergeContext.drawImage(vid, colStart5 + vidWidth * 5, 0, colWidth5, vidHeight, colStart5, 0, colWidth5, vidHeight);

            var arrowLength = 0.06 * vidHeight;
            var arrowheadWidth = 0.025 * vidHeight;
            var arrowheadLength = 0.04 * vidHeight;
            var arrowPosY = vidHeight / 10;
            var arrowWidth = 0.007 * vidHeight;
            var currX_array = pos_array.map(pos => vidWidth * pos);

            for (var i = 0; i < currX_array.length; i++) {
                mergeContext.beginPath();
                mergeContext.arc(currX_array[i], arrowPosY, arrowLength * 0.7, 0, Math.PI * 2, false);
                mergeContext.fillStyle = "#FFD79340";
                mergeContext.fill();

                mergeContext.beginPath();
                mergeContext.moveTo(vidWidth * pos_array[i], 0);
                mergeContext.lineTo(vidWidth * pos_array[i], vidHeight);
                mergeContext.closePath();
                mergeContext.strokeStyle = "#AAAAAA";
                mergeContext.lineWidth = 3;
                mergeContext.stroke();

                mergeContext.beginPath();
                mergeContext.moveTo(currX_array[i], arrowPosY - arrowWidth / 3);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3 - arrowheadLength / 3, arrowPosY - arrowWidth / 3);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3 - arrowheadLength / 3, arrowPosY - arrowheadWidth / 3);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3, arrowPosY);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3 - arrowheadLength / 3, arrowPosY + arrowheadWidth / 3);
                mergeContext.lineTo(currX_array[i] + arrowLength / 3 - arrowheadLength / 3, arrowPosY + arrowWidth / 3);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY + arrowWidth / 3);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY + arrowheadWidth / 3);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3, arrowPosY);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY - arrowheadWidth / 3);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY);
                mergeContext.lineTo(currX_array[i] - arrowLength / 3 + arrowheadLength / 3, arrowPosY - arrowWidth / 3);
                mergeContext.lineTo(currX_array[i], arrowPosY - arrowWidth / 3);
                mergeContext.closePath();
                mergeContext.fillStyle = "#AAAAAA";
                mergeContext.fill();
            }
            requestAnimationFrame(drawLoop);
        }
        requestAnimationFrame(drawLoop);
    }
}

function resizeAndPlay6(element)
{
  var cv = document.getElementById(element.id + "Merge6");
  cv.width = element.videoWidth/6;
  cv.height = element.videoHeight;
  element.play();
  element.style.height = "0px";  // Hide video without stopping it
    
  playVids6(element.id);
}
