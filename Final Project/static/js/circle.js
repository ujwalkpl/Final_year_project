// document.getElementById("myDIV").className = "mystyle";
// stroke-dasharray="50, 100"
// var value = 50;
// $("#chart").attr("data-percent", value.toString());

fetch("sample-file.json")
  .then(Response => Response.json())
  .then(data => {
    console.log(data);
    var percent = data.value.toString()+", 100";
    document.getElementsByClassName('circle')[0].setAttribute("stroke-dasharray", percent);

    if(data.result == "positive"){
      $("#guldu").removeClass('alert alert-dark').addClass('alert alert-danger');
    }
    if(data.result == "negative"){
      $("#guldu").removeClass('alert alert-dark').addClass('alert alert-success');
    }
    
    let newclass = document.querySelector('.report')
    newclass.textContent=data.result;

    let myclass = document.querySelector('.percentage')
    myclass.textContent=data.value.toString()+"%";

    console.log(percent);
    
  }
  
  )
