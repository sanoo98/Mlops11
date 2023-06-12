/* Theme Name:  Tulsy- Responsive Landing page template
  File Description: Main JS file of the template
*/


//  Window scroll sticky class add
function windowScroll() {
  const navbar = document.getElementById("navbar");
  if (
    document.body.scrollTop >= 50 ||
    document.documentElement.scrollTop >= 50
  ) {
    navbar.classList.add("nav-sticky");
  } else {
    navbar.classList.remove("nav-sticky");
  }
}

window.addEventListener('scroll', (ev) => {
  ev.preventDefault();
  windowScroll();
})

// Swiper slider

var swiper = new Swiper(".mySwiper", {
  slidesPerView: 1,
  spaceBetween: 30,
  loop: true,
  loopFillGroupWithBlank: true,
  pagination: {
    el: ".swiper-pagination",
    clickable: true,
  },
  navigation: {
    nextEl: ".swiper-button-next",
    prevEl: ".swiper-button-prev",
  },
});


// swiper testiomonial
var swiper = new Swiper(".mySwiper2", {
  slidesPerView: 1,
  spaceBetween: 0,
  loop: true,
  navigation: {
    nextEl: ".swiper-button-next",
    prevEl: ".swiper-button-prev",
  },
  pagination: {
    el: ".swiper-pagination",
    clickable: true,
  },
  breakpoints: {
    640: {
      slidesPerView: 2,
    },
    768: {
      slidesPerView: 2,
    },
    1024: {
      slidesPerView: 3,
    },
  },
});


// swiper screenshot
var swiper = new Swiper(".mySwiper3", {
  slidesPerView: 1,
  spaceBetween: 0,
  loop: true,
  navigation: {
    nextEl: ".swiper-button-next",
    prevEl: ".swiper-button-prev",
  },
  pagination: {
    el: ".swiper-pagination",
    clickable: true,
  },
  breakpoints: {
    640: {
      slidesPerView: 2,
    },
    768: {
      slidesPerView: 3,
    },
    1024: {
      slidesPerView: 4,
    },
  },
});

// swiper book-landing

var swiper = new Swiper(".mySwiper4", {
  slidesPerView: 1,
  spaceBetween: 0,
  pagination: {
    el: ".swiper-pagination",
    clickable: true,
  },
  breakpoints: {
    640: {
      slidesPerView: 2,
    },
    1024: {
      slidesPerView: 3,
    },
  },
});


// business-slider CLIENT

var swiper = new Swiper(".mySwiper5", {
  slidesPerView: 1,
  spaceBetween: 30,
  loop: true,
  pagination: {
    el: ".swiper-pagination",
    clickable: true,
  },
  navigation: {
    nextEl: ".swiper-button-next",
    prevEl: ".swiper-button-prev",
  },
});

// construction-slider  &  interir-desin slider

var swiper = new Swiper(".mySwiper6", {
  slidesPerView: 2,
  spaceBetween: 0,
  loop: true,
  autoplay: {
    delay: 2000,
  },
  pagination: {
    el: ".swiper-pagination",
    clickable: true,
  },
});

// Swiper slider

var swiper = new Swiper(".mySwiper7", {
  slidesPerView: 4,
  spaceBetween: 0,
  loop: true,
  loopFillGroupWithBlank: true,
  pagination: {
    el: ".swiper-pagination",
    clickable: true,
  },
  navigation: {
    nextEl: ".swiper-button-next",
    prevEl: ".swiper-button-prev",
  },
});


// photographer
try {
  const texts = ["I'M PHOTOGRAPHER", "I'M DESIGNER", "I'M WEB DEVELOPER"];
  var count = 0;
  var index = 0;
  var decrement = 0;
  var currentText = '';
  var letter = '';

  function sleep(delay) {
    return new Promise(resolve => setTimeout(resolve, delay));
  }

  const typeWrite = async () => {
    if (count == texts.length) {
      count = 0;
    }
    currentWord = texts[count];
    currentLetter = currentWord.slice(0, ++index);
    // console.log("document.querySelector", document.getElementsByClassName(".typing2"))
    var typing2 = document.querySelector(".typing2");
    if (typing2) {
      console.log("querySelector", document.querySelector(".typing2"))
      document.querySelector(".typing2").textContent = currentLetter;
      if (index == currentWord.length) {
        await sleep(1500);
        while (index > 0) {
          currentLetter = currentWord.slice(0, --index);
          document.querySelector(".typing2").textContent = currentLetter;
          await sleep(50);
        }
        count++;
        index = 0;
        await sleep(500);
      }
      
      setTimeout(typeWrite, Math.random() * 200 + 50);
    }
    }
    typeWrite();
  // }  
} catch (error) {

}

// software slider

var swiper = new Swiper(".homeslider", {
  effect: 'coverflow',
  loop: true,
  centeredSlides: true,
  slidesPerView: 2,
  initialSlide: 3,
  keyboardControl: true,
  mousewheelControl: true,
  lazyLoading: true,
  preventClicks: false,
  preventClicksPropagation: false,
  lazyLoadingInPrevNext: true,
  grabCursor: true,
  navigation: {
    nextEl: ".swiper-button-next",
    prevEl: ".swiper-button-prev",
  },
  coverflow: {
    rotate: 0,
    depth: 200,
    modifier: 1,
    slideShadows: false,
    slidesPerView: 3,
  }
});

// Collapse Menu 
const navLinks = document.querySelectorAll('.nav-item');
const menuToggle = document.getElementById('navbarCollapse');
let bsCollapse = '';
window.addEventListener('load', function () {
  window.dispatchEvent(new Event('resize'));
});
window.addEventListener('resize', function () {
  var windowSize = document.documentElement.clientWidth;
  bsCollapse = new bootstrap.Collapse(menuToggle, { toggle: false });
  if (windowSize < 980) {
    navLinks.forEach((l) => {
      l.addEventListener('click', () => { toggleMenu(); });
    });
  } else {
    toggleMenu();
  }
});

function toggleMenu() {
  var windowSize = document.documentElement.clientWidth;
  if (windowSize < 980) {
    bsCollapse.toggle();
  } else {
    bsCollapse = '';
  }
}
