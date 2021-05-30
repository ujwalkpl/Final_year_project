// document.getElementById("myDIV").className = "mystyle";
// stroke-dasharray="50, 100"
// var value = 50;
// $("#chart").attr("data-percent", value.toString());

fetch(covid)
  .then(Response => Response.json())
  .then(data => {
    console.log(data);
    var percent = data.value.toString()+", 100";
    document.getElementsByClassName('circle')[0].setAttribute("stroke-dasharray", percent);

    let myclass = document.querySelector('.percentage')
    myclass.textContent=data.value.toString()+"%";

    console.log(percent);

  }
  
  )
